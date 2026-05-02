using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Networking;
using TMPro;


public class Song_UI_shower : MonoBehaviour
{
    [Header("certain song")]
    public Song song;
    [Header("UI Setting")]
    public Image targetImage;
    public Image BackImage;
    public TextMeshProUGUI title;
    public TextMeshProUGUI sub_title;

    [Header("Others Setting")]
    public bool is_first = false; // 這個為第一個才需要勾選的 bool，讓一開始就觸發選擇的功能

    [Header("imag setting private")]
    private string cover_image_path;
    private SongSelectMenu SSM;

    [Header("audio setting private")]
    private string audio_path;
    private AudioSource audioSource;  // 存現在播放的內容
    private AudioSource nextAudioSource;  // 下一個播放的內容
    // Start is called before the first frame update
    void Start()
    {
        SSM = SongSelectMenu.instance;
        audioSource = SSM.audioSource;
        nextAudioSource = SSM.nextAudioSource;
        string fullPath = Path.Combine(Application.streamingAssetsPath, song.dir, song.info_name);
        StartCoroutine(StreamingAssetLoader.ReadFileFromJar(fullPath, (json) => {
            if (json != null)
            {
                SongInfo songInfo = JsonUtility.FromJson<SongInfoWrapper>(json).ToSongInfo();
                string img_relative_path = Path.Combine(song.dir, songInfo._coverImageFilename);
                string audio_relative_path = Path.Combine(song.dir, songInfo._songFilename);

                audio_path = Path.Combine(Application.streamingAssetsPath, song.dir, songInfo._songFilename);

                cover_image_path = Path.Combine(Application.streamingAssetsPath, song.dir, songInfo._coverImageFilename);

                StartCoroutine(StreamingAssetLoader.LoadImage(cover_image_path, (sprite) => {
                    if (sprite != null)
                    {
                        targetImage.sprite = sprite;
                        if (is_first)
                        {
                            song_selected();
                        }
                    }
                }));
                title.text = song.name;
                sub_title.text = song.description;
            }
        }));

    }

    // Update is called once per frame
    void Update()
    {
        
    }

    public void song_selected()
    {
        if(SSM.last_selected_song == this) return;
        Debug.Log("############################ selected song: " + song.name);
        play_audio();

        change_img_color(BackImage, "#CE4760"); // 轉成紅色 // selected
        if (!is_first) change_img_color(SSM.last_selected_song.BackImage, "#1B5174"); // 轉成藍色 // default


        SSM.reset_fill_position(); // reset level fill
        // 設定右側的內容
        if (!is_first) SSM.targetImage.sprite = targetImage.sprite;
        SSM.title.text = song.name;
        SSM.sub_title.text = song.description;
        SSM.levelData.level = "MI"; // 預設第一個選項
        // 紀錄上一個選的
        SSM.last_selected_song = this;
        song.SaveSettings();   // 存檔
        SSM.song = song;
        // 顯示 對應 level 根據 SO song 
        ManageObjects(song.ui_show_names, SSM.level_parent_obj);
        is_first = false;
    }

    public void ManageObjects(string[] names, GameObject p_obj)
    {
        foreach (string name in names)
        {
            // 嘗試在 PObject 下尋找名稱為 name 的物件
            Transform child = p_obj.transform.Find(name);

            if (child != null)
            {
                // 找到物件，設置為 Active
                child.gameObject.SetActive(true);
            }
            else
            {
                // 沒有找到物件，創建新的並設置為 Active
                GameObject newObject = new GameObject(name);
                newObject.transform.SetParent(p_obj.transform);
                newObject.SetActive(true);
            }
        }

        // 確保 PObject 下未在 names 陣列中的物件被隱藏
        foreach (Transform child in p_obj.transform)
        {
            if (!System.Array.Exists(names, name => name == child.gameObject.name))
            {
                child.gameObject.SetActive(false);
            }
        }
    }

    public string FormatTime(float totalSeconds) // 秒數轉換分鐘，並顯示像是 3:20 這種格式
    {
        int minutes = Mathf.FloorToInt(totalSeconds / 60f);
        int seconds = Mathf.FloorToInt(totalSeconds % 60f);
        return string.Format("{0}:{1:00}", minutes, seconds);
    }
    void change_img_color(Image img, string hexColor)
    {
        // 將十六進位顏色轉換成 Color 類型
        if (ColorUtility.TryParseHtmlString(hexColor, out Color newColor))
        {
            // 更新 Image 顏色
            img.color = newColor;
        }
    }

    void play_audio()
    {
        StartCoroutine(StreamingAssetLoader.LoadAudio(audio_path, 0, AudioType.OGGVORBIS, (clip) => {
            if (clip != null)
            {
                audioSource.clip = clip;
                audioSource.Play();
                // 設定右側的內容 如果不寫在這裡，會讀取到上首歌曲的內容
                SSM.song_time.text = "Song Time: " + FormatTime(audioSource.clip.length).ToString();
                SSM.get_note_num(0);
            }
        }));
        /*if (audioSource == null)
        {
            StartCoroutine(StreamingAssetLoader.LoadAudio(audio_path, 0, AudioType.OGGVORBIS, (clip) => {
                if (clip != null)
                {
                    audioSource.clip = clip;
                    audioSource.Play();
                }
            }));
        }
        else
        {
            PlayNewSong(audio_path, 0, AudioType.OGGVORBIS);
        }*/
    }

    /*// 播放新歌曲並實現平滑過渡
    public void PlayNewSong(string path, float delay_time, AudioType type)
    {
        StartCoroutine(SmoothTransition(path, delay_time, type));
    }

    private IEnumerator SmoothTransition(string newSongPath, float delay_time, AudioType type)
    {
        // 減小當前歌曲音量
        float fadeOutDuration = 1f; // 1秒內減小音量
        float fadeInDuration = 2f;  // 2秒內增大音量

        // 播放新歌曲，並載入
        yield return StartCoroutine(StreamingAssetLoader.LoadAudio(newSongPath, delay_time, type, (newClip) => {
            nextAudioSource.clip = newClip;
            nextAudioSource.Play();
        }));

        // 減小當前歌曲的音量
        float startVolume = audioSource.volume;
        float targetVolume = 0f;
        float elapsedTime = 0f;
        while (elapsedTime < fadeOutDuration)
        {
            audioSource.volume = Mathf.Lerp(startVolume, targetVolume, elapsedTime / fadeOutDuration);
            elapsedTime += Time.deltaTime;
            yield return null;
        }
        audioSource.volume = targetVolume;
        audioSource.Stop();

        // 增大新歌曲音量
        nextAudioSource.volume = 0f;
        elapsedTime = 0f;
        while (elapsedTime < fadeInDuration)
        {
            nextAudioSource.volume = Mathf.Lerp(0f, 1f, elapsedTime / fadeInDuration);
            elapsedTime += Time.deltaTime;
            yield return null;
        }
        nextAudioSource.volume = 1f;

        // 切換到新歌曲
        audioSource = nextAudioSource;
        // nextAudioSource = gameObject.AddComponent<AudioSource>();
    }*/
    // for android
    /*private IEnumerator ReadFileFromJar(string path)
    {
        using (UnityWebRequest www = UnityWebRequest.Get(path))
        {
            yield return www.SendWebRequest();

            if (www.result == UnityWebRequest.Result.Success)
            {
                string json = www.downloadHandler.text;
                // string fullPath = Path.Combine(Application.streamingAssetsPath, song.dir, song.info_name);
                // string json = File.ReadAllText(fullPath);
                SongInfo songInfo = JsonUtility.FromJson<SongInfoWrapper>(json).ToSongInfo();

                /*string img_relative_path = Path.Combine(song.dir, songInfo._coverImageFilename);
                string audio_relative_path = Path.Combine(song.dir, songInfo._songFilename);
                //string json_relative_path = Path.Combine(song.dir, song.info_name);


                //StartCoroutine(StreamingAssetLoader.CopyToPersistentDataPath(img_relative_path, (path) => {
                //    Debug.Log("已複製到：" + path);
                //}));
               // StartCoroutine(StreamingAssetLoader.CopyToPersistentDataPath(audio_relative_path, (path) => {
                //    Debug.Log("已複製到：" + path);
                //}));
               // StartCoroutine(StreamingAssetLoader.CopyToPersistentDataPath(json_relative_path, (path) => {
               //     Debug.Log("已複製到：" + path);
               // }));

                audio_path = Path.Combine(Application.streamingAssetsPath, song.dir, songInfo._songFilename);

                cover_image_path = Path.Combine(Application.streamingAssetsPath, song.dir, songInfo._coverImageFilename);



                StartCoroutine(StreamingAssetLoader.LoadImage(cover_image_path, (sprite) => {
                    if (sprite != null)
                    {
                        targetImage.sprite = sprite;
                    }
                }));
                title.text = song.name;
                sub_title.text = song.description;

                if (is_first)
                {
                    song_selected();
                }
            }
            else
            {
                Debug.LogError("Failed to read file: " + www.error);
            }
        }
    }*/
}
