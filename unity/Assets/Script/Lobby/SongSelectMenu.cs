using System.Collections;
using System.Collections.Generic;
using System.IO;
using TMPro;
using UnityEngine;
using UnityEngine.UI;
using Newtonsoft.Json;
using System.Threading;
[System.Serializable]
public class LevelData
{
    public string level;
}
public class SongSelectMenu : MonoBehaviour
{
    public static SongSelectMenu instance;
    [Header("selected song")]
    public Song song;
    [Header("Main Menu Buttons")]
    public Button startButton;
    [Header("use in Song UI shower")]
    public Song_UI_shower last_selected_song;
    public GameObject level_parent_obj; // 讓 Song UI shower 可以設定難度 (用 child set Active)
    public Image targetImage;
    public TextMeshProUGUI title;
    public TextMeshProUGUI sub_title;
    public TextMeshProUGUI song_time;
    public TextMeshProUGUI cube_num;
    [Header("use in Song UI shower")]
    public GameObject fill_obj;  // 為了移動 fill 的位置，讓使用者可以看到目前選擇的難度
    private Vector3 fill_obj_pos;  // 為了移動 fill 的位置，讓使用者可以看到目前選擇的難度
    public AudioSource audioSource;  // 存現在播放的內容 // 在 Song_UI_shower.cs 切換
    public AudioSource nextAudioSource;  // 下一個播放的內容 // 在 Song_UI_shower.cs 切換 // 目前沒有用到
    public LevelData levelData = new LevelData { level = "MI" };  // 紀錄遊戲模式，在 Song_UI_shower select 的時候會設定為 MI，改模式在下方的 button 觸發會改成 button 名稱
    private TCP_Client sender;

    void Awake()
    {
        if (instance == null)
            instance = this;
        else
        {
            Destroy(gameObject);
            return;
        }
        audioSource.volume = Config.volume;  
        // DontDestroyOnLoad(gameObject);
        fill_obj_pos = fill_obj.GetComponent<RectTransform>().localPosition;
    }
    
    // Start is called before the first frame update
    void Start()
    {
        //Hook events
        startButton.onClick.AddListener(StartGame);
        save_game_level();
        sender = TCP_Client.instance;
    }

    // 當按鈕被點擊時觸發，並將 Button 傳遞給方法 // 主要為 中間選擇 level 的 button 使用
    public void OnLevelButtonClick(Button clickedButton)
    {
        // 獲取按鈕的位置
        Vector3 buttonPosition = clickedButton.transform.localPosition;
        levelData.level = clickedButton.gameObject.name;

        int index = GetObjectsIdbyName(levelData.level, level_parent_obj);
        get_note_num(index);

        // 將 UI 元素移動到按鈕的位置 // 手動調整
        RectTransform uiRectTransform = fill_obj.GetComponent<RectTransform>();
        Vector3 offset = new Vector3(84, -51, 0);
        uiRectTransform.localPosition = buttonPosition + offset;
        save_game_level();
    }

    public void save_game_level()
    {
        string json = JsonUtility.ToJson(levelData);
        string fullPath = Path.Combine(Application.persistentDataPath, Config.level_json_file);
        File.WriteAllText(fullPath, json);
    }
    public void reset_fill_position()
    {
        fill_obj.GetComponent<RectTransform>().localPosition = fill_obj_pos;
    }

    public void StartGame()
    {
        if(levelData.level == Config.level_ui_strings[(int)Level.MI])
        {
            sender.send_string_to_python(song.name);

            if (song.name == Config.Calibration) GameDataManager.instance.is_calibration = true;    

            SceneLoaderManager.instance.LoadMIStage();
            /*if (song.name == Config.Calibration) SceneLoaderManager.instance.LoadCalibrationStage();
            else SceneLoaderManager.instance.LoadMIStage();*/
        }
        else
        {
            SceneLoaderManager.instance.LoadBeatSaberStage();
        }
    }
    int GetObjectsIdbyName(string names, GameObject p_obj)
    {
        int count = 0;
        // 確保 PObject 下未在 names 陣列中的物件被隱藏
        foreach (Transform child in p_obj.transform)
        {
            if (child.gameObject.activeSelf)
            {
                if (names == child.name) break;
                count += 1;
            }
        }
        return count;
    }
    public void get_note_num(int index) // index 代表中間的那個，選擇難度，讀取看對應的 note 數量
    {
        string filePath = Path.Combine(Application.streamingAssetsPath, song.dir, song.file_names[index]);
        StartCoroutine(StreamingAssetLoader.ReadFileFromJar(filePath, (json) => {
            if (json != null)
            {
                OldBeatmap beatmap = JsonConvert.DeserializeObject<OldBeatmap>(json);
                if (beatmap == null || beatmap.notes == null)
                {
                    Debug.LogError("Failed to deserialize .dat file or _notes is null");
                    if (index == 0) cube_num.text = "MI trial: 0";
                    else cube_num.text = "Cubes: 0";
                }
                // if (index == 0) cube_num.text = "MI trial: " + beatmap.notes.Count.ToString();
                // if (song.name == Config.Calibration) cube_num.text = "MI trial: 20";
                // else if (index == 0) cube_num.text = "MI trial: " + ((int)(beatmap.notes.Count / (Config.group_note_num + 1)) + 1).ToString();
                if (song.name == Config.Calibration) cube_num.text = "MI trial: 40";
                else if (index == 0) cube_num.text = "MI trial: " + ((int)(beatmap.notes.Count / (Config.group_note_num))).ToString();  // 40
                else cube_num.text = "Cubes: " + beatmap.notes.Count.ToString();

                /*BeatmapData data = JsonConvert.DeserializeObject<BeatmapData>(json);
                if (data == null || data.colorNotes == null)  // use old beat map format
                {
                    OldBeatmap beatmap = JsonConvert.DeserializeObject<OldBeatmap>(json);
                    if (beatmap == null || beatmap.notes == null)
                    {
                        Debug.LogError("Failed to deserialize .dat file or _notes is null");
                        if (index == 0) cube_num.text = "MI trial: 0";
                        else cube_num.text = "Cubes: 0";
                    }
                    if (index == 0) cube_num.text = "MI trial: " + beatmap.notes.Count.ToString();
                    else cube_num.text = "Cubes: " + beatmap.notes.Count.ToString();
                }
                else
                {
                    if (index == 0) cube_num.text = "MI trial: " + data.colorNotes.Count.ToString();
                    else cube_num.text = "Cubes: " + data.colorNotes.Count.ToString();
                }*/
            }
        }));
    }
    // 把 BeatmapSpawner.cs 裡面的 GroupNotesByRules，function 做更改，改成只回傳 int 的 function，並於畫面中顯示，目前沒用到
    public int GetGroupCountByRules(List<OldNote> sourceNotes, int n, float bpm)
    {
        float beatsPerSecond = bpm / 60f;

        float beatThreshold_up = (0.6f * Config.group_note_num + 2) * beatsPerSecond;
        float beatThreshold_down = (0.6f * Config.group_note_num) * beatsPerSecond;
        float beatThreshold_1s = 1.0f * beatsPerSecond;
        // 如果沒有資料，回傳 0
        if (sourceNotes == null || sourceNotes.Count == 0) return 0;
        if (n <= 0) n = 1;

        // --- 條件 1: 如果 n = 1，每個 Note 都是一組，直接回傳總數 ---
        if (n == 1)
        {
            return sourceNotes.Count;
        }

        int groupCount = 0; // 用來計算多少 group
        int currentIndex = 0;
        int totalNotes = sourceNotes.Count;

        while (currentIndex < totalNotes)
        {
            List<OldNote> currentGroup = new List<OldNote>();
            OldNote firstNote = sourceNotes[currentIndex];
            currentGroup.Add(firstNote);

            // --- 嘗試湊滿 n 個，同時檢查上限限制 ---
            int lookAheadCount = 1;
            while (lookAheadCount < n && (currentIndex + lookAheadCount) < totalNotes)
            {
                OldNote nextNote = sourceNotes[currentIndex + lookAheadCount];
                float timeDiff = nextNote.time - firstNote.time;
                if (timeDiff > beatThreshold_up) break;

                // Debug.Log("Diff: " + timeDiff);

                currentGroup.Add(nextNote);
                lookAheadCount++;
            }

            // --- 條件 3: 檢查是否過短需要丟棄 ---
            bool shouldAddGroup = true;
            if (currentGroup.Count < n)
            {
                float groupDuration = currentGroup[currentGroup.Count - 1].time - currentGroup[0].time;
                // 如果不足 n 且時間差太短 -> 丟棄
                if (groupDuration < beatThreshold_down)
                {
                    shouldAddGroup = false;
                }
            }

            // --- 更新索引與應用新規則 ---
            if (shouldAddGroup)
            {
                // [修改] 不用加入 List，直接計數器 + 1
                groupCount++;

                // 預設：跳過 Group 數量 + 1 (刪除後面那一個 Note)
                int skipAmount = currentGroup.Count + 1;

                // 檢查：下一個 Note 是否離這一組的最後一個 Note 太遠 (> 1秒)
                int nextNoteIndex = currentIndex + currentGroup.Count;

                // 確保沒有超出範圍
                if (nextNoteIndex < totalNotes)
                {
                    OldNote lastNoteInGroup = currentGroup[currentGroup.Count - 1];
                    OldNote nextNoteToCheck = sourceNotes[nextNoteIndex];

                    float gap = nextNoteToCheck.time - lastNoteInGroup.time;

                    // 如果間隔超過 1 秒 -> 改為「不刪除」
                    // 也就是只跳過 currentGroup.Count，保留 nextNoteToCheck 給下一輪
                    if (gap > beatThreshold_1s)
                    {
                        skipAmount = currentGroup.Count;
                    }
                }

                // 執行索引更新
                currentIndex += skipAmount;
            }
            else
            {
                // 如果這組被丟棄了，跳過這組的 Note，不需要額外再刪除後面一個
                currentIndex += currentGroup.Count;
            }
        }
        // 回傳計算出來的數量
        return groupCount;
    }
}
