using System;
#if LCK_FMOD_2_03
#define LCK_FMOD
#endif
using System.Collections.Generic;
using Liv.Lck.NativeMicrophone;
using UnityEngine;
using Liv.Lck.Settings;
using Unity.Profiling;
using Liv.Lck.Collections;
#if PLATFORM_ANDROID && !UNITY_EDITOR
using UnityEngine.Android;
#endif

namespace Liv.Lck
{
    internal class LckAudioMixer : ILckAudioMixer, ILckLateUpdate
    {
        private ILckAudioSource _gameAudioSource;
        private bool _isGameAudioMuted = false;
        private float _gameAudioGain = 1.0f;
        private Queue<float> _gameAudioQueue = new Queue<float>();

        private ILckAudioSource _nativeMicrophoneCapture;
        private bool _isMicrophoneMuted = false;
        private float _microphoneGain = 1.0f;
        private Queue<float> _microphoneQueue = new Queue<float>();

        private AudioBuffer _micAudioBuffer = new AudioBuffer(96000);
        private float _lastMicrophoneLevel;

        private AudioBuffer _gameAudioBuffer = new AudioBuffer(96000);
        private float _lastGameAudioLevel;

        private AudioBuffer _mixedAudioBuffer = new AudioBuffer(96000);
        private int _remainingGameAudioValuesToAdjust;
        private int _gameAudioValueCountOffset;
        
        static readonly ProfilerMarker _lateUpdateProfileMarker = new ProfilerMarker("LckAudioMixer.LateUpdate");
        private readonly int _sampleRate;
        private Component _audioCaptureMarker;

        private const int _targetAudioBufferLength = 1024;

        private ILckAudioLimiter _lckAudioLimiterHard;
        private ILckAudioLimiter _lckAudioLimiterSoft;
        private ILckAudioLimiter _lckAudioLimiterCurve;

        private const int TrackTimeDifferenceToleranceMilli = 100;
        private const int NumberOfChannels = 2;

        private float? _micCaptureStartRecordingTime;
        private int _totalMicSamples;
        private int _totalGameSamples;
        
        public LckAudioMixer(int sampleRate)
        {
            _sampleRate = sampleRate;

            VerifyAudioCaptureComponent();

            _nativeMicrophoneCapture = new LckNativeMicrophone(_sampleRate);

            var settings = AudioSettings.GetConfiguration();

            LckUpdateManager.RegisterSingleLateUpdate(this);
            
            _lckAudioLimiterHard = new LckHardLimiter(threshold: 0.65f, ratio: 6f);
            _lckAudioLimiterSoft = new LckSoftLimiter(threshold: 0.6f, kneeWidth: 0.8f, ratio: 12f);
        }

        public AudioBuffer GetMixedAudio(float recordingTime)
        {
            return MixAudioArrays(recordingTime);
        }
        
        public void EnableCapture()
        {
            VerifyAudioCaptureComponent();

            if(_gameAudioSource != null)
            {
                _gameAudioSource.EnableCapture();

                _micCaptureStartRecordingTime = null;
                _microphoneQueue.Clear();
                _gameAudioQueue.Clear();

                _totalGameSamples = 0;
                _totalMicSamples = 0;
                
                // HACK: This is a workaround for the delay we are observing in the game audio.
                //       Ideally we resolve this delay at its source
                //

                // Calculate the number of samples to adjust for game audio
                var gameAudioTimeOffsetSeconds = LckSettings.Instance.GameAudioSyncTimeOffsetInMS / 1000f;
                _gameAudioValueCountOffset = Mathf.CeilToInt(gameAudioTimeOffsetSeconds * _sampleRate) * NumberOfChannels;
                _remainingGameAudioValuesToAdjust = _gameAudioValueCountOffset;
            }
        }

        public void DisableCapture()
        {
            if(_gameAudioSource != null)
            {
                _gameAudioSource.DisableCapture();
                _micCaptureStartRecordingTime = null;
                _microphoneQueue.Clear();
                _gameAudioQueue.Clear();
            }
        }
        
        public static int GetSampleRate()
        {
#if LCK_WWISE
            return (int)AkSoundEngine.GetSampleRate();
#elif LCK_FMOD
            FMODUnity.RuntimeManager.CoreSystem.getSoftwareFormat(out int sampleRate, out _, out _);
            return sampleRate;
#endif

#if LCK_NOT_UNITY_AUDIO
            return LckSettings.Instance.FallbackSampleRate;
#else
            return AudioSettings.outputSampleRate;
#endif
        }
        
        private AudioBuffer MixAudioArrays(float recordingTime)
        {
            if(_gameAudioSource == null)
            {
                LckLog.LogError("LCK No game audio source found");
                return null;
            }
            
            var shouldIncludeMicAudio = _nativeMicrophoneCapture.IsCapturing();
            
            // Keep track of the recording time when the mic started capturing so that we can use it to monitor whether
            // the microphone audio data is keeping up with the expected sample rate and make adjustments accordingly
            if (!_micCaptureStartRecordingTime.HasValue && shouldIncludeMicAudio)
                _micCaptureStartRecordingTime = recordingTime;
            
            // Enqueue samples from audio buffers into queues ready to be mixed
            EnqueueGameBufferSamples();

            if (shouldIncludeMicAudio)
            {
                EnqueueMicBufferSamples();
            }
            else
            {
                _microphoneQueue.Clear();
            }

            // Ensure the number of samples in each audio source queue is about what we'd expect, or adjust accordingly
            // (This prevents audio sources getting out of sync if one or both are not meeting the expected sample rate)
            EnsureAudioSourceSamplesWithinTolerance(nameof(_gameAudioSource), recordingTime, _gameAudioQueue, ref _totalGameSamples);
            if (shouldIncludeMicAudio && _micCaptureStartRecordingTime.HasValue)
            {
                var micRecordingTime = recordingTime - _micCaptureStartRecordingTime.Value;
                EnsureAudioSourceSamplesWithinTolerance(nameof(_nativeMicrophoneCapture), micRecordingTime, _microphoneQueue,
                    ref _totalMicSamples);
            }
            
            // Mix as much audio as we have available from each audio source queue into the mixed audio buffer
            var availableAudioBlocks = DetermineAvailableBlockCount(shouldIncludeMicAudio);
            return MixBlocksIntoMixedAudioBuffer(shouldIncludeMicAudio, availableAudioBlocks);
        }
        
        private void EnqueueGameBufferSamples()
        {
            // Apply game audio offset adjustment to ensure game audio is in sync with microphone
            // TODO: Investigate why this is needed and see if it can be resolved at its source
            if (_remainingGameAudioValuesToAdjust > 0)
            {
                var valuesToAdd = Math.Max(_remainingGameAudioValuesToAdjust - _gameAudioQueue.Count, 0);
                for (var i = 0; i < valuesToAdd; i++)
                {
                    _gameAudioQueue.Enqueue(0f);
                }
                _remainingGameAudioValuesToAdjust -= valuesToAdd;
            }
            
            // Enqueue samples from game audio buffer
            _totalGameSamples += _gameAudioBuffer.Count / NumberOfChannels;
            for(var i = 0; i < _gameAudioBuffer.Count; i++)
            {
                _gameAudioQueue.Enqueue(_gameAudioBuffer[i] * _gameAudioGain * (_isGameAudioMuted ? 0.0f : 1.0f));
            }
            
            // Apply game audio offset adjustment to ensure game audio is in sync with microphone
            // TODO: Investigate why this is needed and see if it can be resolved at its source
            if (_remainingGameAudioValuesToAdjust < 0)
            {
                var valuesToRemove = Mathf.Min(Mathf.Abs(_remainingGameAudioValuesToAdjust), _gameAudioQueue.Count);
                for (var i = 0; i < valuesToRemove; i++)
                {
                    _gameAudioQueue.Dequeue();
                }
            
                _remainingGameAudioValuesToAdjust += valuesToRemove;
            }
        }

        private void EnqueueMicBufferSamples()
        {
            if (_micAudioBuffer == null)
                return;
            
            // Enqueue samples from mic audio buffer
            _totalMicSamples += _micAudioBuffer.Count / NumberOfChannels;
            for (var i = 0; i < _micAudioBuffer.Count; i++)
            {
                _microphoneQueue.Enqueue(_micAudioBuffer[i] * _microphoneGain *
                                         (_isMicrophoneMuted ? 0.0f : 1.0f));
            }
        }
        
        private int DetermineAvailableBlockCount(bool shouldIncludeMicAudio)
        {
            var availableGameBlocks = CountAvailableGameBlocks();
            if (!shouldIncludeMicAudio)
                return availableGameBlocks;

            var availableMicBlocks = CountAvailableMicrophoneBlocks();
            return Mathf.Min(availableGameBlocks, availableMicBlocks);
        }
        
        private int CountAvailableGameBlocks()
        {
            // Leave enough samples to cover the game audio offset in the queue to avoid providing audio too early
            var availableGameAudioValues = _gameAudioQueue.Count;
            
            if (_gameAudioValueCountOffset > 0)
            {
                // Leave enough samples to cover the game audio offset in the queue to avoid providing audio too early
                availableGameAudioValues -= _gameAudioValueCountOffset;
            }
            
            var gameBufferBlocks = Math.Max(0, availableGameAudioValues / _targetAudioBufferLength);
            return gameBufferBlocks;
        }

        private int CountAvailableMicrophoneBlocks()
        {
            var availableMicAudioValues = _microphoneQueue.Count;
            if (_gameAudioValueCountOffset < 0)
            {
                // Leave enough samples to cover the game audio offset in the queue to avoid providing audio too early
                availableMicAudioValues += _gameAudioValueCountOffset;
            }
            
            var micBufferBlocks = Math.Max(0, availableMicAudioValues / _targetAudioBufferLength);
            return micBufferBlocks;
        }
        
        private AudioBuffer MixBlocksIntoMixedAudioBuffer(bool shouldIncludeMicAudio, int blocks)
        {
            var outputLength = blocks * _targetAudioBufferLength;
            _mixedAudioBuffer.Clear();
            
            for(var i = 0; i < outputLength; i++)
            {
                var mixedAudioRaw = _gameAudioQueue.Dequeue();
                
                if (shouldIncludeMicAudio)
                {
                    mixedAudioRaw += _microphoneQueue.Dequeue();
                }
                
                var finalMixedAudio = ApplyLimiter(mixedAudioRaw);
                if(!_mixedAudioBuffer.TryAdd(finalMixedAudio))
                {
                    LckLog.LogWarning("LCK Mixed audio buffer overflow");
                    break;
                }
            }

            return _mixedAudioBuffer;
        }

        private float ApplyLimiter(float mixedAudioRaw)
        {
            var finalMixedAudio = mixedAudioRaw;

            switch (LckSettings.Instance.AudioLimiter)
            {
                case LckSettings.LimiterType.SoftClip:
                    finalMixedAudio = LckAudioLimiterUtils.ApplySoftClip(mixedAudioRaw);
                    break;
                case LckSettings.LimiterType.None:
                    finalMixedAudio = mixedAudioRaw;
                    break;
            }

            return finalMixedAudio;
        }

        public void LateUpdate()
        {
            using (_lateUpdateProfileMarker.Auto())
            {
                if(!VerifyAudioCaptureComponent())
                {
                    return;
                }

                if (_nativeMicrophoneCapture.IsCapturing())
                {
                    _nativeMicrophoneCapture.GetAudioData(MicrophoneAudioDataCallback);
                }

                _gameAudioSource.GetAudioData(GameAudioDataCallback);
            }
        }

        private void MicrophoneAudioDataCallback(AudioBuffer audioBuffer)
        {
            _micAudioBuffer.Clear();

            if(audioBuffer.Count > 0)
            {
                if(!_micAudioBuffer.TryCopyFrom(audioBuffer))
                {
                    LckLog.LogError("LCK Mic audio data copy failed");
                    return;
                }

                _lastMicrophoneLevel = (_lastMicrophoneLevel + CalculateRootMeanSquare(_micAudioBuffer)) / 2.0f;
            }
        }

        private void GameAudioDataCallback(AudioBuffer audioBuffer)
        {
            _gameAudioBuffer.Clear();

            if(audioBuffer.Count > 0)
            {
                if(!_gameAudioBuffer.TryCopyFrom(audioBuffer))
                {
                    LckLog.LogError("LCK Game audio data copy failed");
                    return;
                }
                                
                _lastGameAudioLevel = (_lastGameAudioLevel + CalculateRootMeanSquare(audioBuffer)) / 2.0f;
                
                if (float.IsNaN(_lastGameAudioLevel))
                    _lastGameAudioLevel = 0;
            }
        }

        private bool VerifyAudioCaptureComponent()
        {
            if (_audioCaptureMarker == null)
            {
                var allListeners = LckMonoBehaviourMediator.FindObjectsOfType<AudioListener>(false);
                var activeListeners = new List<AudioListener>();

                for (var index = 0; index < allListeners.Length; index++)
                {
                    var listener = allListeners[index];
                    if (listener.enabled)
                    {
                        activeListeners.Add(listener);
                    }
                }

                if (activeListeners.Count == 0)
                {
                    LckLog.Log("LCK Found no audio listener in the scene, looking for AudioCaptureMarker");

                    var markers = LckMonoBehaviourMediator.FindObjectsOfType<LckAudioMarker>(false);

                    if (markers.Length > 0)
                    {
                        _audioCaptureMarker = markers[0];
                    }

                    if (markers.Length > 1)
                    {
                        LckLog.LogError("LCK found more than one AudioCaptureMarker in the scene. This is not valid");
                    }
                }
                else
                {
                    if (activeListeners.Count > 0)
                    {
                        _audioCaptureMarker = activeListeners[0];
                    }

                    if (activeListeners.Count > 1)
                    {
                        LckLog.LogError("LCK found more than one active AudioListener in the scene. This is not valid");
                    }
                }
            }

            if (_gameAudioSource == null)
            {
                _gameAudioSource = _audioCaptureMarker.gameObject.GetComponent<ILckAudioSource>();

                if (_gameAudioSource == null)
                {
#if LCK_FMOD
                    _gameAudioSource = _audioCaptureMarker.gameObject.AddComponent<LckAudioCaptureFMOD>();
#elif LCK_WWISE
                    _gameAudioSource = _audioCaptureMarker.gameObject.AddComponent<LckAudioCaptureWwise>();
#else
                    _gameAudioSource = _audioCaptureMarker.gameObject.AddComponent<LckAudioCapture>();
#endif
                }

            }

            return true;
        }

        private bool CheckMicAudioPermissions()
        {
#if PLATFORM_ANDROID && !UNITY_EDITOR
            return Permission.HasUserAuthorizedPermission(Permission.Microphone);
#else
            return true;
#endif
        }

        public LckResult SetMicrophoneCaptureActive(bool active)
        {
            _lastMicrophoneLevel = 0;

            if (!CheckMicAudioPermissions())
            {
                return LckResult.NewError(LckError.MicrophonePermissionDenied,
                    "The app has not been granted microphone permissions.");
            }

            if (active)
            {
                _nativeMicrophoneCapture.EnableCapture();
            }
            else
            {
                _nativeMicrophoneCapture.DisableCapture();
                _micCaptureStartRecordingTime = null;
            }
            
            _totalMicSamples = 0;

            return LckResult.NewSuccess();
        }

        public LckResult<bool> GetMicrophoneCaptureActive()
        {
            return LckResult<bool>.NewSuccess(_nativeMicrophoneCapture.IsCapturing());
        }

        public LckResult SetGameAudioMute(bool isMute)
        {
            _isGameAudioMuted = isMute;

            return LckResult.NewSuccess();
        }
        
        public LckResult<bool> IsGameAudioMute()
        {
            return LckResult<bool>.NewSuccess(_isGameAudioMuted);
        }

        public void SetMicrophoneGain(float gain)
        {
            _microphoneGain = gain;
        }

        public void SetGameAudioGain(float gain)
        {
            _gameAudioGain = gain;
        }

        public float GetMicrophoneOutputLevel()
        {
            return _lastMicrophoneLevel;
        }
        
        public float GetGameOutputLevel()
        {
            return _lastGameAudioLevel;
        }

        private static float CalculateRootMeanSquare(AudioBuffer audioBuffer)
        {
            if (audioBuffer == null || audioBuffer.Count == 0)
            {
                return 0;
            }

            float sum = 0;
            for (int i = 0; i < audioBuffer.Count; i++)
            {
                sum += audioBuffer[i] * audioBuffer[i];
            }

            return Mathf.Sqrt(sum / audioBuffer.Count);
        }

        private static void PadWithSilence(Queue<float> audioQueue, int samplesToAdd, ref int runningSampleCount)
        {
            for (var sampleIdx = 0; sampleIdx < samplesToAdd; sampleIdx++)
            {
                for (var channel = 0; channel < NumberOfChannels; channel++)
                {
                    audioQueue.Enqueue(0f);
                }
                
                runningSampleCount++;
            }
        }
        
        private void EnsureAudioSourceSamplesWithinTolerance(string audioSourceName, float captureTime, 
            Queue<float> audioSourceQueue, ref int audioSourceRunningSampleCount)
        {
            var expectedSampleCount = Mathf.FloorToInt(captureTime * _sampleRate);
            var sampleCountDifference = audioSourceRunningSampleCount - expectedSampleCount;
            var absSampleCountDifference = Math.Abs(sampleCountDifference);
            var sampleCountDifferenceTolerance = TrackTimeDifferenceToleranceMilli * (_sampleRate / 1000);
            if (absSampleCountDifference <= sampleCountDifferenceTolerance)
                return;
            
            if (sampleCountDifference < 0)
            {
                // This may occur if the audio thread is not able to keep up with target sample rate
                LckLog.LogWarning($"{audioSourceName} is behind expected sample count ({expectedSampleCount}) by " + 
                                  $"{absSampleCountDifference} samples - Padding with silence for missing samples");
                
                // To get back in sync with video and other audio sources, pad the audio queue with silence
                PadWithSilence(audioSourceQueue, absSampleCountDifference, ref audioSourceRunningSampleCount);
            }
            else
            {
                LckLog.LogWarning($"{audioSourceName} is ahead of expected sample count ({expectedSampleCount}) by " + 
                               $"{absSampleCountDifference} samples - Expecting this to be a result of a lag spike");
            }
        }
        
        public void Dispose()
        {
            LckUpdateManager.UnregisterSingleLateUpdate(this);

            // TODO: This is not ideal. Could possibly add IDIsposible to ILckAudioSource?
            ( _nativeMicrophoneCapture as LckNativeMicrophone ).Dispose();
        }
    }
}
