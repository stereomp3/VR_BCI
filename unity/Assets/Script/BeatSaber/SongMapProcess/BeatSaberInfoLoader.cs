using UnityEngine;
using System.IO;
using UnityEngine.UI;
using System.Collections.Generic;
using System.Collections;
using UnityEngine.Networking;
using Meta.Voice.Hub;
using OVRSimpleJSON;

public class BeatSaberInfoLoader : MonoBehaviour
{
    [Header("UI Setting (no use)")]
    public Text songNameText;
    public Text authorNameText;
    public Image coverImage;

    [Header("Song Setting")]
    public Song song;
    public bool use_json_to_load = false;
    public bool is_end_song = false; // 為了讓下面的 continue 不要在歌曲結束的時候透過暫停開始歌曲，並且讓 GM set final acc 只觸發一次
    [Header("Audio Setting")]
    public AudioSource audioSource;
    public bool play_audio = true;
    public float play_delay_seconds_offs = 0f; // 不知為什麼時間無法對上，所以人工調整，這邊和 forward.cs 裡面設定最後位置和時間位移會有關連
    private float play_delay_seconds = 2.5f; // 從 GM 拿，GM.song_delay_time - play_delay_seconds_offset
    public float song_bpm = 120;

    private GameManager GM;
    private string coverPath;
    private void Awake()
    {
        if (use_json_to_load) song.LoadSettings();
        string fullPath = Path.Combine(Application.streamingAssetsPath, song.dir, song.info_name);

        StartCoroutine(StreamingAssetLoader.ReadFileFromJar(fullPath, (json) => {
            if (json != null)
            {
                SongInfo songInfo = JsonUtility.FromJson<SongInfoWrapper>(json).ToSongInfo();
                song_bpm = songInfo._beatsPerMinute;
                // Display on canvas
                //songNameText.text = songInfo._songName;
                //authorNameText.text = songInfo._songAuthorName;
                Debug.Log("songInfo._songName @@@@@@@@@@@@@@@@@@@:" + songInfo._songName);
                Debug.Log("songInfo._songAuthorName @@@@@@@@@@@@@@@@@@@:" + songInfo._songAuthorName);
                Debug.Log("songInfo._beatsPerMinute @@@@@@@@@@@@@@@@@@@:" + songInfo._beatsPerMinute);
                Debug.Log("songInfo._songFilename @@@@@@@@@@@@@@@@@@@:" + songInfo._songFilename);

                coverPath = Path.Combine(Application.streamingAssetsPath, song.dir, songInfo._songFilename);
                // Use songInfo._difficultyBeatmapSets as needed
                foreach (var set in songInfo._difficultyBeatmapSets)
                {
                    Debug.Log($"Characteristic: {set._beatmapCharacteristicName}");
                    foreach (var diff in set._difficultyBeatmaps)
                    {
                        Debug.Log($"  {diff._difficulty} - Rank {diff._difficultyRank}, File: {diff._beatmapFilename}");
                    }
                }
                StartCoroutine(LoadSongAudio());
            }
        }));

    }
    private void Start()
    {
        
    }

    void pause_song()
    {
        audioSource.Pause();
    }

    public void continue_song()
    {
        if (is_end_song) return;
        audioSource.Play();
    }

    private IEnumerator LoadSongAudio()
    {
        GM = GameManager.instance;
        GM.onGameStopCallback += pause_song;
        GM.onGameStartCallback += continue_song;

        play_delay_seconds = GM.song_delay_time - play_delay_seconds_offs + GM.start_game_seconds; // 不知為什麼時間無法對上，所以人工調整
        /*StartCoroutine(StreamingAssetLoader.LoadImage(coverPath, (sprite) => {
            if (sprite != null) coverImage.sprite = sprite;
        }));*/

        yield return StreamingAssetLoader.LoadAudio(coverPath, play_delay_seconds, AudioType.OGGVORBIS, (clip) =>
        {
            if (clip != null)
            {
                audioSource.clip = clip;
                audioSource.Play();
            }
        }
    );
    }
}