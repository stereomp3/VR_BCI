// WAVUtility.cs
using UnityEngine;
using System;
using System.Text;

public static class WAVUtility
{
    // ±N WAV bytes ¸ÑªR¬° Unity AudioClip
    public static AudioClip ToAudioClip(byte[] wavFileBytes, string clipName = "wav")
    {
        try
        {
            // find "data" chunk
            int pos = 12; // skip RIFF header
            while (pos < wavFileBytes.Length - 4)
            {
                string chunkId = Encoding.ASCII.GetString(wavFileBytes, pos, 4);
                int chunkSize = BitConverter.ToInt32(wavFileBytes, pos + 4);
                if (chunkId.ToLower() == "data")
                {
                    pos += 8;
                    int dataSize = chunkSize;
                    // Read format info
                    int channels = BitConverter.ToInt16(wavFileBytes, 22);
                    int sampleRate = BitConverter.ToInt32(wavFileBytes, 24);
                    int bitsPerSample = BitConverter.ToInt16(wavFileBytes, 34);
                    int bytesPerSample = bitsPerSample / 8;

                    int numSamplesPerChannel = dataSize / (bytesPerSample * channels);
                    float[] floatData = new float[numSamplesPerChannel * channels];

                    int offset = pos;
                    int i = 0;
                    if (bitsPerSample == 16)
                    {
                        for (int s = 0; s < numSamplesPerChannel; s++)
                        {
                            for (int c = 0; c < channels; c++)
                            {
                                short sample = BitConverter.ToInt16(wavFileBytes, offset + (s * channels + c) * 2);
                                floatData[i++] = sample / 32768f;
                            }
                        }
                    }
                    else if (bitsPerSample == 8)
                    {
                        for (int s = 0; s < numSamplesPerChannel; s++)
                        {
                            for (int c = 0; c < channels; c++)
                            {
                                byte sample = wavFileBytes[offset + s * channels + c];
                                floatData[i++] = (sample - 128) / 128f;
                            }
                        }
                    }
                    else
                    {
                        Debug.LogError("Unsupported bits per sample: " + bitsPerSample);
                        return null;
                    }

                    AudioClip audioClip = AudioClip.Create(clipName, numSamplesPerChannel, channels, sampleRate, false);
                    audioClip.SetData(floatData, 0);
                    return audioClip;
                }
                else
                {
                    pos += 8 + chunkSize;
                }
            }
            Debug.LogError("Could not find data chunk in WAV");
            return null;
        }
        catch (Exception e)
        {
            Debug.LogError("WAV parse error: " + e);
            return null;
        }
    }
}