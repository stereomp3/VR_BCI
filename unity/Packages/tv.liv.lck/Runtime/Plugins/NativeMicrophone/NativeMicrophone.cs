using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using Liv.Lck.Collections;
using Liv.Lck.Settings;
using UnityEngine;

namespace Liv.Lck.NativeMicrophone
{
    public enum LogLevel : uint
    {
        Off,
        Error,
        Warn,
        Info,
        Debug,
        Trace,
    }

    public class LckNativeMicrophone : IDisposable, ILckAudioSource
    {
        public enum ReturnCode : uint
        {
            Ok = 0,
            Error = 1,
            InvalidKey = 2,
            DefaultInputDeviceError = 3,
            BuildStreamError = 4,
            NoAudioData = 5,
            LoggerAlreadySet = 6,
            CaptureNotStarted = 7,
        }

        private static Dictionary<UInt64, LckNativeMicrophone> _instances = new Dictionary<UInt64, LckNativeMicrophone>();

        private const string __DllName = "native_microphone";

        private UInt64 _nativeInstance;
        private AudioDataCallbackDelegate _callback;

        private AudioBuffer _audioBuffer = new AudioBuffer(96000);

        private IntPtr _callbackPtr;
        private bool _isCapturing;
        private bool _shouldDisableCapture;
        private bool _shouldEnableCapture;
        private Task _setMicStateTask;

        [DllImport(__DllName, CallingConvention = CallingConvention.Cdecl)]
        private static extern UInt64 microphone_capture_new(UInt32 sampleRate);

        [DllImport(__DllName, CallingConvention = CallingConvention.Cdecl)]
        private static extern ReturnCode microphone_capture_free(UInt64 audioCaptureKey);

        [DllImport(__DllName, CallingConvention = CallingConvention.Cdecl)]
        private static extern ReturnCode microphone_capture_start(UInt64 audioCaptureKey);

        [DllImport(__DllName, CallingConvention = CallingConvention.Cdecl)]
        private static extern ReturnCode microphone_capture_stop(UInt64 audioCaptureKey);

        [DllImport(__DllName, CallingConvention = CallingConvention.Cdecl)]
        private static extern ReturnCode microphone_capture_get_audio(
            UInt64 audioCaptureKey,
            IntPtr callback
        );

        [DllImport(__DllName, CallingConvention = CallingConvention.Cdecl)]
        private static extern void set_max_log_level(LogLevel levelFilter);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate void AudioDataCallbackDelegate(IntPtr dataPtr, int length, UInt64 audioCaptureKey);

        public LckNativeMicrophone(int sampleRate)
        {
            SetMaxLogLevel(LckSettings.Instance.MicrophoneLogLevel);

            _callback = AudioDataCallback;
            _callbackPtr = Marshal.GetFunctionPointerForDelegate(_callback);
            _nativeInstance = microphone_capture_new((uint)sampleRate);
            _instances.Add(_nativeInstance, this);
        }

        [AOT.MonoPInvokeCallback(typeof(AudioDataCallbackDelegate))]
        private static void AudioDataCallback(IntPtr dataPtr, int length, UInt64 audioCaptureKey)
        {
            if(_instances.TryGetValue(audioCaptureKey, out var instance))
            {
                try
                {
                    if (instance._audioBuffer.Capacity < length)
                    {
                        LckLog.LogWarning($"LCK Native Microphone dropping audio: {instance._audioBuffer.Capacity} < {length}");
                    }
                    var countToCopy = Mathf.Min(length, instance._audioBuffer.Capacity);
                    if (!instance._audioBuffer.TryCopyFrom(dataPtr, countToCopy))
                    {
                        LckLog.LogError($"LCK Mic Audio data copy failed");
                        return;
                    }
                }
                catch (Exception ex)
                {
                    Debug.LogError($"LCK Exception during mic audio copy: {ex.Message}");
                }
            }
            else
            {
                LckLog.LogError("LCK NativeMicrophone: Could not find instance for key: " + audioCaptureKey);
            }
        }

        public void Dispose()
        {
            microphone_capture_free(_nativeInstance);
            _instances.Remove(_nativeInstance);
        }

        public bool IsCapturing()
        {
            return _isCapturing;
        }

        public void GetAudioData(ILckAudioSource.AudioDataCallbackDelegate callback)
        {
            _audioBuffer.Clear();

            if (_isCapturing)
            {
                var result = ReturnCode.Error;
                result = microphone_capture_get_audio(_nativeInstance, _callbackPtr);
            }

            callback(_audioBuffer);
        }

        public void EnableCapture()
        {
            _shouldEnableCapture = true;
            _shouldDisableCapture = false;

            if (_setMicStateTask == null)
                _setMicStateTask = Task.Run(() => SetMicrophoneCaptureActive(true));
        }

        public void DisableCapture()
        {
            _shouldDisableCapture = true;
            _shouldEnableCapture = false;

            if (_setMicStateTask == null)
                _setMicStateTask = Task.Run(() => SetMicrophoneCaptureActive(false));
        }

        private async Task SetMicrophoneCaptureActive(bool active)
        {
            if (_isCapturing == active)
            {
                _setMicStateTask = null;
                return;
            }

            if (active)
            {
                _isCapturing = microphone_capture_start(_nativeInstance) == ReturnCode.Ok;
                _shouldEnableCapture = false;

                _setMicStateTask = null;
                if (_shouldDisableCapture)
                {
                    await SetMicrophoneCaptureActive(false);
                }
            }
            else
            {
                _isCapturing = false;
                microphone_capture_stop(_nativeInstance);
                _shouldDisableCapture = false;

                _setMicStateTask = null;
                if (_shouldEnableCapture)
                {
                    await SetMicrophoneCaptureActive(true);
                }
            }
        }

        public static void SetMaxLogLevel(LogLevel logLevel)
        {
            set_max_log_level(logLevel);
        }
    }
}
