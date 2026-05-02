using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Networking;
using System.Collections;
using System.Collections.Generic;
using System;
using System.IO;

public static class StreamingAssetLoader
{
    /// <summary>
    /// 根據平台取得正確的 StreamingAssets 路徑 URL。
    /// </summary>
    private static string GetPlatformURL(string path)
    {
        #if UNITY_ANDROID && !UNITY_EDITOR
                // Android 的 path 已經是 jar:file:// 開頭，不可再加 file://
                return path;
        #else   // if (Application.platform == RuntimePlatform.Android)
                // 其他平台需要加上 file://
                if (!path.StartsWith("file://"))
                    return "file://" + path;
                return path;
        #endif
    }
    // 載入圖片：給定完整路徑，回傳 Sprite
    public static IEnumerator LoadImage(string path, System.Action<Sprite> onLoaded)
    {
        // string url = "file://" + path;
        string url = GetPlatformURL(path);
        using (UnityWebRequest uwr = UnityWebRequestTexture.GetTexture(url))
        {
            yield return uwr.SendWebRequest();

            if (uwr.result != UnityWebRequest.Result.Success)
            {
                Debug.LogError("fail to load sprite: " + uwr.error);
                onLoaded?.Invoke(null);
            }
            else
            {
                // 取得載入的圖片
                Texture2D texture = DownloadHandlerTexture.GetContent(uwr);

                // 計算裁剪後的尺寸
                int targetSize = Mathf.Min(texture.width, texture.height);
                int startX = (texture.width - targetSize) / 2;
                int startY = (texture.height - targetSize) / 2;

                // 從中間裁剪出 1:1 比例的部分
                Color[] croppedPixels = texture.GetPixels(startX, startY, targetSize, targetSize);

                // 創建新的 Texture2D 存儲裁剪後的圖片
                Texture2D croppedTexture = new Texture2D(targetSize, targetSize);
                croppedTexture.SetPixels(croppedPixels);
                croppedTexture.Apply();

                // 將裁剪後的 Texture 轉為 Sprite 並返回
                Sprite croppedSprite = Sprite.Create(
                    croppedTexture,
                    new Rect(0, 0, croppedTexture.width, croppedTexture.height),
                    new Vector2(0.5f, 0.5f)
                );

                // 執行回調
                onLoaded?.Invoke(croppedSprite);
            }
        }
    }
    public static IEnumerator ReadFileFromJar(string path, System.Action<string> onLoaded)
    {
        using (UnityWebRequest uwr = UnityWebRequest.Get(path))
        {
            yield return uwr.SendWebRequest();

            if (uwr.result == UnityWebRequest.Result.Success)
            {
                string json = uwr.downloadHandler.text;
                // 執行回調
                onLoaded?.Invoke(json);
            }
            else
            {
                Debug.LogError("Failed to read json: " + uwr.error);
                onLoaded?.Invoke(null);
            }
        }
    }
    // 載入音訊：給定完整路徑，回傳 AudioClip
    public static IEnumerator LoadAudio(string path, float delay_time, AudioType type, System.Action<AudioClip> onLoaded)
    {
        // string url = "file://" + path;
        string url = GetPlatformURL(path);
        using (UnityWebRequest uwr = UnityWebRequestMultimedia.GetAudioClip(url, type))
        {
            yield return uwr.SendWebRequest();

            if (uwr.result != UnityWebRequest.Result.Success)
            {
                Debug.LogError("fail to load audio: " + uwr.error);
                onLoaded?.Invoke(null);
            }
            else
            {
                AudioClip clip = DownloadHandlerAudioClip.GetContent(uwr);
                yield return new WaitForSeconds(delay_time); // wait for start
                onLoaded?.Invoke(clip);
            }
        }
    }

    /// <summary>
    /// 將 StreamingAssets 中的檔案複製到 persistentDataPath。
    /// 適合想使用普通 File IO 存取時使用。
    /// </summary>
    public static IEnumerator CopyToPersistentDataPath(string relativePath, Action<string> onCopied = null)
    {
        string sourcePath = Path.Combine(Application.streamingAssetsPath, relativePath);
        string destPath = Path.Combine(Application.persistentDataPath, relativePath);
        string url = GetPlatformURL(sourcePath);
        if (!File.Exists(destPath))
        {
            string dir = Path.GetDirectoryName(destPath);
            if (!Directory.Exists(dir))
                Directory.CreateDirectory(dir);

            using (UnityWebRequest uwr = UnityWebRequest.Get(url))
            {
                yield return uwr.SendWebRequest();

                if (uwr.result != UnityWebRequest.Result.Success)
                {
                    Debug.LogError($"[StreamingAssetLoader] 複製失敗: {uwr.error}\nURL: {url}");
                }
                else
                {
                    File.WriteAllBytes(destPath, uwr.downloadHandler.data);
                    onCopied?.Invoke(destPath);
                }
            }
        }
        else
        {
            onCopied?.Invoke(destPath + ", path exist.");
        }
    }
    // for android
    private static IEnumerator ReadFileFromJar(string path)
    {
        using (UnityWebRequest www = UnityWebRequest.Get(path))
        {
            yield return www.SendWebRequest();

            if (www.result == UnityWebRequest.Result.Success)
            {
                string jsonContent = www.downloadHandler.text;
                Debug.Log(jsonContent);
            }
            else
            {
                Debug.LogError("Failed to read file: " + www.error);
            }
        }
    }
    /*/// <summary>
    /// 根據平台取得正確的 StreamingAssets 路徑 URL。
    /// </summary>
    private static string GetPlatformURL(string path)
    {
#if UNITY_ANDROID && !UNITY_EDITOR
        // Android 的 path 已經是 jar:file:// 開頭，不可再加 file://
        return path;
#else
        // 其他平台需要加上 file://
        if (!path.StartsWith("file://"))
            return "file://" + path;
        return path;
#endif
    }

    // ✅ 載入圖片（可用於 StreamingAssets）
    public static IEnumerator LoadImage(string path, Action<Sprite> onLoaded)
    {
        string url = GetPlatformURL(path);

        using (UnityWebRequest uwr = UnityWebRequestTexture.GetTexture(url))
        {
            yield return uwr.SendWebRequest();

            if (uwr.result != UnityWebRequest.Result.Success)
            {
                Debug.LogError($"[StreamingAssetLoader] 圖片載入失敗: {uwr.error}\nURL: {url}");
                onLoaded?.Invoke(null);
                yield break;
            }

            Texture2D texture = DownloadHandlerTexture.GetContent(uwr);

            // 取得中間 1:1 裁切圖像
            int targetSize = Mathf.Min(texture.width, texture.height);
            int startX = (texture.width - targetSize) / 2;
            int startY = (texture.height - targetSize) / 2;
            Color[] croppedPixels = texture.GetPixels(startX, startY, targetSize, targetSize);

            Texture2D croppedTexture = new Texture2D(targetSize, targetSize);
            croppedTexture.SetPixels(croppedPixels);
            croppedTexture.Apply();

            Sprite croppedSprite = Sprite.Create(
                croppedTexture,
                new Rect(0, 0, croppedTexture.width, croppedTexture.height),
                new Vector2(0.5f, 0.5f)
            );

            onLoaded?.Invoke(croppedSprite);
        }
    }

    // ✅ 載入音訊（可用於 StreamingAssets）
    public static IEnumerator LoadAudio(string path, float delayTime, AudioType type, Action<AudioClip> onLoaded)
    {
        string url = GetPlatformURL(path);

        using (UnityWebRequest uwr = UnityWebRequestMultimedia.GetAudioClip(url, type))
        {
            yield return uwr.SendWebRequest();

            if (uwr.result != UnityWebRequest.Result.Success)
            {
                Debug.LogError($"[StreamingAssetLoader] 音訊載入失敗: {uwr.error}\nURL: {url}");
                onLoaded?.Invoke(null);
                yield break;
            }

            AudioClip clip = DownloadHandlerAudioClip.GetContent(uwr);
            yield return new WaitForSeconds(delayTime);
            onLoaded?.Invoke(clip);
        }
    }
    /// <summary>
    /// 將 StreamingAssets 中的檔案複製到 persistentDataPath。
    /// 適合想使用普通 File IO 存取時使用。
    /// </summary>
    public static IEnumerator CopyToPersistentDataPath(string relativePath, Action<string> onCopied = null)
    {
        string sourcePath = Path.Combine(Application.streamingAssetsPath, relativePath);
        string destPath = Path.Combine(Application.persistentDataPath, relativePath);
        string url = GetStreamingAssetURL(sourcePath);

        if (!File.Exists(destPath))
        {
            string dir = Path.GetDirectoryName(destPath);
            if (!Directory.Exists(dir))
                Directory.CreateDirectory(dir);

            using (UnityWebRequest uwr = UnityWebRequest.Get(url))
            {
                yield return uwr.SendWebRequest();

                if (uwr.result != UnityWebRequest.Result.Success)
                {
                    Debug.LogError($"[StreamingAssetLoader] 複製失敗: {uwr.error}\nURL: {url}");
                }
                else
                {
                    File.WriteAllBytes(destPath, uwr.downloadHandler.data);
                    onCopied?.Invoke(destPath);
                }
            }
        }
        else
        {
            onCopied?.Invoke(destPath);
        }
    }
     */

}

// Unity's JsonUtility doesn't support top-level lists or dictionaries well,
// so we need a wrapper class to deserialize properly.
[System.Serializable]
public class SongInfoWrapper // BeatSaberInfoLoader 也有
{
    public string _songName;
    public string _songAuthorName;
    public string _coverImageFilename;
    public string _songFilename;
    public float _beatsPerMinute;
    public float _songTimeOffset;
    public float _previewStartTime;
    public float _previewDuration;
    public List<DifficultyBeatmapSet> _difficultyBeatmapSets;

    public SongInfo ToSongInfo()
    {
        return new SongInfo
        {
            _songName = _songName,
            _songAuthorName = _songAuthorName,
            _coverImageFilename = _coverImageFilename,
            _songFilename = _songFilename,
            _beatsPerMinute = _beatsPerMinute,
            _songTimeOffset = _songTimeOffset,
            _previewStartTime = _previewStartTime,
            _previewDuration = _previewDuration,
            _difficultyBeatmapSets = _difficultyBeatmapSets
        };
    }
}