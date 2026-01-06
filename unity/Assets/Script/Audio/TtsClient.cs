using System.Collections;
using UnityEngine;
using UnityEngine.Networking;
using System.Text;
using System;

public class TtsClient : MonoBehaviour
{
    #region Singleton 
    public static TtsClient instance;
    void Awake()
    {
        if (instance != null)
        {
            Debug.LogWarning("More than one instance of TtsClient found!");
            Destroy(gameObject);
            return;
        }
        else
        {
            // DontDestroyOnLoad(gameObject);
            instance = this;
        }
    }

    #endregion
    public string serverUrl = "http://127.0.0.1:5000/speak";
    public AudioSource audioSource;

    [Serializable]
    class TtsRequest { public string text; }

    public void Speak(string text)
    {
        StartCoroutine(SpeakCoroutine(text));
    }

    IEnumerator SpeakCoroutine(string text)
    {
        var reqObj = new TtsRequest { text = text };
        string json = JsonUtility.ToJson(reqObj);
        byte[] bodyRaw = Encoding.UTF8.GetBytes(json);

        using (UnityWebRequest www = new UnityWebRequest(serverUrl, "POST"))
        {
            www.uploadHandler = new UploadHandlerRaw(bodyRaw);
            www.downloadHandler = new DownloadHandlerBuffer();
            www.SetRequestHeader("Content-Type", "application/json");

            yield return www.SendWebRequest();

            if (www.result != UnityWebRequest.Result.Success)
            {
                Debug.LogError("TTS request failed: " + www.error);
                yield break;
            }

            byte[] audioBytes = www.downloadHandler.data;
            if (audioBytes == null || audioBytes.Length == 0)
            {
                Debug.LogError("Empty audio bytes");
                yield break;
            }

            // ¸ÑªR WAV bytes ¦¨ AudioClip
            AudioClip clip = WAVUtility.ToAudioClip(audioBytes, "tts_clip");
            if (clip == null)
            {
                Debug.LogError("Failed to parse WAV");
                yield break;
            }

            audioSource.clip = clip;
            audioSource.Play();
        }
    }
}