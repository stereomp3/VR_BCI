using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json;
using UnityEngine.UIElements;

public class BeatmapSpawner : MonoBehaviour
{   // 要把 run 的訓練用回來，需要改 TimeLogger.cs 裡面傳送到 python 那邊，以及 ECEOCondition
    public GameObject blockPrefab_red;
    public GameObject blockPrefab_blue;
    public GameObject LinePrefab_red;
    public GameObject LinePrefab_blue;
    public float bpm = 120f;
    public float cude_space = 1.5f;
    public bool use_random = false;
    public float spawn_y_offset = 0f;
    protected int seed = 42;

    protected GameManager GM;
    protected float songStartTime;
    protected float songDelayTime = 0f; // 減去暫停時間
    protected bool is_pause = false;
    // protected List<ColorNote> notes = new List<ColorNote>();
    protected List<OldNote> old_notes = new List<OldNote>();
    protected HashSet<OldNote> finalNote = new HashSet<OldNote>(); // 用來記錄每一組最後一個 Note (用於腦波控制)
    protected Dictionary<OldNote, float> noteGroupDurationMap = new Dictionary<OldNote, float>();  // 用來記錄每個 Note 所屬 Group 的時間差 (LastTime - FirstTime)

    protected int currentNoteIndex = 0;
    protected BeatSaberInfoLoader beatSaberInfoLoader;  // require other script
    protected float per_spawn_second = 2.5f;  // 會事先產生 cube，然後到定點時間出現在指定位置
    protected float start_game_seconds = 0f;
    protected bool is_sent_line = true;  // 設定一個 group，會出現一個 line
    protected Song song;
    protected float beatsPerSecond;
    protected float beatThreshold_up; // 創建 group 的最常容許秒數
    protected float beatThreshold_down;  // 創建 group 的最低容許秒數
    protected float beatThreshold_1s;  // 1 秒
    protected TCP_Client TC;  // 用於傳送睜眼閉眼

    void Start()
    {
        GM = GameManager.instance;
        GM.onGameStopCallback += stop_gmae;
        GM.onGameStartCallback += start_gmae;

        TC = TCP_Client.instance;

        beatSaberInfoLoader = GetComponent<BeatSaberInfoLoader>(); // select difficulty
        song = beatSaberInfoLoader.song;
        bpm = beatSaberInfoLoader.song_bpm;
        beatsPerSecond = bpm / 60f;

        per_spawn_second = GM.song_delay_time;
        start_game_seconds = GM.start_game_seconds;

        load_song();

        songStartTime = Time.time;
    }
    
    void Update()
    {
        if (is_pause)
        {
            songDelayTime += Time.deltaTime;
        }
        else
        {
            if (start_game_seconds < 0)
            {
                float elapsedTime = Time.time - songStartTime - songDelayTime;
                float currentBeat = elapsedTime * beatsPerSecond;
                if (currentNoteIndex >= old_notes.Count)
                {
                    if (!beatSaberInfoLoader.is_end_song)
                    {
                        StartCoroutine(GM.SetFinalAcc());
                        // StartCoroutine(ECEOCondition(Config.adaptive_model)); // 睜眼閉眼任務，與是否訓練
                        StartCoroutine(ECEOCondition(false)); // 睜眼閉眼任務，與是否訓練
                    }
                    beatSaberInfoLoader.is_end_song = true;
                    return;
                }

                while (currentNoteIndex < old_notes.Count && old_notes[currentNoteIndex].time <= currentBeat)
                {
                    SpawnOldBlock(old_notes[currentNoteIndex]);
                    if (GM.is_audosaber)
                    {
                        int x = old_notes[currentNoteIndex].lineIndex, y = old_notes[currentNoteIndex].lineLayer, d = old_notes[currentNoteIndex].cutDirection;
                        if (GM.onAutoSaberCallback != null)
                        {
                            if (finalNote.Contains(old_notes[currentNoteIndex]) || Config.group_note_num <= 1) // group 小於等於 1，就直接全部腦波控制
                            {
                                StartCoroutine(GM.onAutoSaberCallback.Invoke(x, y, d, true));
                                Debug.Log("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ use brain to controll saber!!!");
                                is_sent_line = true;
                            }
                            else
                            {
                                StartCoroutine(GM.onAutoSaberCallback.Invoke(x, y, d, false));
                                if (is_sent_line)
                                {
                                    SpawnLine(old_notes[currentNoteIndex], beatThreshold_up);
                                    is_sent_line = false;   
                                }
                            }
                        }
                        if (GM.setupAutoSaberCallback != null) GM.setupAutoSaberCallback.Invoke();
                    }
                    currentNoteIndex++;
                }
            }
            else
            {
                start_game_seconds -= Time.deltaTime;
                songStartTime = Time.time;
            }
        }
    }
    protected IEnumerator ECEOCondition(bool is_training) // 睜眼閉眼任務
    {
        yield return new WaitForSeconds(10f);
        text2speech.instance.Speak("Please Close your eyes.");
        TC.send_string_to_python($"Close eyes: {Timer.GetUnixTimestamp():F3}");
        // yield return new WaitForSeconds(2.5f);
        if (is_training) GM.start_training(); // 根據設定判斷是否要訓練
        yield return new WaitForSeconds(20f);
        
        text2speech.instance.Speak("Please Open your eyes.");
        TC.send_string_to_python($"Open eyes: {Timer.GetUnixTimestamp():F3}");
        yield return new WaitForSeconds(20f);
        text2speech.instance.Speak("This run of testing has ended.");
    }

    protected void load_song()
    {
        beatThreshold_up = (Config.cube_space_time * Config.group_note_num + 2) * beatsPerSecond;
        beatThreshold_down = (Config.cube_space_time * Config.group_note_num) * beatsPerSecond;
        beatThreshold_1s = 1.0f * beatsPerSecond;

        // string json = File.ReadAllText(filePath);
        string filePath = Path.Combine(Application.streamingAssetsPath, song.dir, song.file_names[0]);
        StartCoroutine(StreamingAssetLoader.ReadFileFromJar(filePath, (json) => {
            if (json != null)
            {
                OldBeatmap beatmap = JsonConvert.DeserializeObject<OldBeatmap>(json);
                if (beatmap == null || beatmap.notes == null)
                {
                    Debug.LogError("Failed to deserialize .dat file or _notes is null");
                    return;
                }
                if (use_random) beatmap = set_random_old_note(beatmap, Config.group_note_num, beatThreshold_up, beatThreshold_down, beatThreshold_1s);
                old_notes = beatmap.notes;
                GM.correct_total = old_notes.Count * 1;  // tmp // 把 group 結果給到 GM
            }
        }));
    }
    protected void SpawnOldBlock(OldNote note)
    {
        Vector3 position = new Vector3(note.lineIndex - 1.5f, note.lineLayer + spawn_y_offset, 0) * cude_space; // Adjust as needed
        // new Vector3(0, 0.3f, 0); 
        Quaternion rotation = Quaternion.identity;
        switch (note.cutDirection)
        {
            case 0: rotation = Quaternion.Euler(0, 0, 0); break;      // Up
            case 1: rotation = Quaternion.Euler(0, 0, 180); break;    // Down
            case 2: rotation = Quaternion.Euler(0, 0, 270); break;    // Left
            case 3: rotation = Quaternion.Euler(0, 0, 90); break;     // Right
            case 4: rotation = Quaternion.Euler(0, 0, 315); break;    // Up-Left
            case 5: rotation = Quaternion.Euler(0, 0, 45); break;     // Up-Right
            case 6: rotation = Quaternion.Euler(0, 0, 225); break;    // Down-Left
            case 7: rotation = Quaternion.Euler(0, 0, 135); break;    // Down-Right
            case 8: rotation = Quaternion.identity; break;           // Any
        }
        GameObject obj;
        if (note.type == 1) obj = Instantiate(blockPrefab_blue, position + transform.position, rotation);
        else obj = Instantiate(blockPrefab_red, position + transform.position, rotation);

        // StartCoroutine(AnimateNote(obj));
        // Optional: Rotate or animate based on direction
        obj.name = $"Note [t={note.time:F2}, dir={note.cutDirection}]";
        if(Config.group_note_num == 1) obj.AddComponent<NoteLogTrigger>();  // 只有一個 note 就要加入這個才會有 start 和 end
    }
    protected void SpawnLine(OldNote note, float beatThreshold_up, float assign_dur=-1) // 生成線條，用於 group start end，之後也可以單看 cut，然後由 cut 往前或是後推，SaberSlicer 砍下去會送出 CUT
    {
        Vector3 position = new Vector3(note.lineIndex - 1.5f, note.lineLayer - 0.3f + spawn_y_offset, -1.5f) * cude_space; // Adjust as needed
        Quaternion rotation = Quaternion.identity;
        GameObject lineObj;
        if (note.type == 1) lineObj = Instantiate(LinePrefab_blue, position + transform.position, rotation);
        else lineObj = Instantiate(LinePrefab_red, position + transform.position, rotation);

        // 2. 從字典取得這組的時間差 (note_last_time - note_first_time)
        float groupDuration = 0f;
        if (noteGroupDurationMap.TryGetValue(note, out float duration))
        {
            groupDuration = duration;
        }
        if (assign_dur != -1)  // 這段是為了在 calibration 自記創建 note 的時候，會沒有 duration，所以需要自己設定
        {
            groupDuration = assign_dur; 
        }
        // 3. 計算比例並修改 Scale X
        if (beatThreshold_up > 0) // 防呆避免除以 0
        {
            float scaleRatio = groupDuration / beatThreshold_up;

            // 取得原本的 Scale
            Vector3 newScale = lineObj.transform.localScale;

            // 將 X 軸乘以比例
            newScale.z *= scaleRatio;

            // 應用新的 Scale
            lineObj.transform.localScale = newScale;

            // Debug.Log($"SpawnLine: Duration={groupDuration}, Max={beatThreshold_up}, Ratio={scaleRatio}, NewScaleX={newScale.x}");
        }
    }
    IEnumerator AnimateNote(GameObject note)  // 衝出去動畫，要延遲可以把 Vector3.forward 乘小的值 (調整從後面多少衝到 spawn object)，然後 duration 上升
    {
        float duration = 1.5f;  // 0.5f
        Vector3 start = note.transform.position + Vector3.forward * 300f;  // 100 f
        Vector3 end = note.transform.position;

        float t = 0;
        while (t < duration)
        {
            t += Time.deltaTime;
            note.transform.position = Vector3.Lerp(start, end, t / duration);
            yield return null;
        }
    }
    
    List<int> GenerateRandomBinaryList(int N, int numZeros, int seed) // 從 EEGtrain copy 過來的 (之前傳統的 MI)
    {
        if (numZeros < 0 || numZeros > N)
        {
            Debug.LogError("numZeros 必須介於 0 到 N 之間");
            return null;
        }

        int numOnes = N - numZeros;

        // 初始化 List
        List<int> binaryList = new List<int>();

        for (int i = 0; i < numZeros; i++) binaryList.Add(0);
        for (int i = 0; i < numOnes; i++) binaryList.Add(1);

        // 使用 seed 初始化亂數產生器
        System.Random rng = new System.Random(seed);

        // Fisher-Yates Shuffle
        for (int i = binaryList.Count - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            int temp = binaryList[i];
            binaryList[i] = binaryList[j];
            binaryList[j] = temp;
        }

        return binaryList;
    }
    List<int> GenerateGroupedBinaryList(int N, int groupSize, int seed)
    {
        // 檢查 N 是否能被 groupSize 整除
        if (N % groupSize != 0)
        {
            Debug.LogError($"N ({N}) 必須能被 groupSize ({groupSize}) 整除。");
            return null;
        }

        // 檢查 groupSize 是否為偶數 (為了讓 0 和 1 數量相等)
        if (groupSize % 2 != 0)
        {
            Debug.LogError($"groupSize ({groupSize}) 必須是偶數，才能平均分配 0 和 1。");
            return null;
        }

        // 初始化最終的 List
        List<int> finalBinaryList = new List<int>();

        // 使用 seed 初始化亂數產生器 (只需初始化一次，確保整個序列受控)
        System.Random rng = new System.Random(seed);

        // 計算總共有幾組
        int numGroups = N / groupSize;
        int halfGroup = groupSize / 2;

        // 開始分組產生
        for (int g = 0; g < numGroups; g++)
        {
            // 1. 建立該組的暫存 List
            List<int> groupList = new List<int>();

            // 加入一半的 0
            for (int i = 0; i < halfGroup; i++) groupList.Add(0);
            // 加入一半的 1
            for (int i = 0; i < halfGroup; i++) groupList.Add(1);

            // 2. 對該組進行 Fisher-Yates Shuffle (組內洗牌)
            for (int i = groupList.Count - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                int temp = groupList[i];
                groupList[i] = groupList[j];
                groupList[j] = temp;
            }

            // 3. 將洗牌後的該組加入最終 List
            finalBinaryList.AddRange(groupList);
        }

        return finalBinaryList;
    }

    protected OldBeatmap set_random_old_note(OldBeatmap data) // data 為 單一 list，沒有分組 用於之前 Caliration 系列的，現在 Calibriatino 也用 這個 script
    {
        seed = Random.Range(0, 10000); // 設定隨機種子，在 use_random 為 true，設定隨機左右才會用到
        int numNotes = data.notes.Count;
        int numZeros = numNotes / 2; // 一半 0 的數量
        List<int> binaryList = GenerateRandomBinaryList(numNotes, numZeros, seed);
        // 根據 binaryList 修改 notes
        for (int i = 0; i < data.notes.Count; i++)
        {
            if (binaryList[i] == 0) // right blue
            {
                data.notes[i].lineIndex = 2;
                data.notes[i].type = 1;
            }
            else // left red
            {
                data.notes[i].lineIndex = 1;
                data.notes[i].type = 0;
            }
        }
        return data;
    }
    protected OldBeatmap set_random_old_note(OldBeatmap data, int n, float beatThreshold_up, float beatThreshold_down, float beatThreshold_1s) // data 有分組
    {
        finalNote.Clear();
        noteGroupDurationMap.Clear();

        // 1. 根據的規則進行分組
        List<List<OldNote>> groups = GroupNotesByRules(data.notes, n, beatThreshold_up, beatThreshold_down, beatThreshold_1s);

        /*// 2. 如果 Group 數量是奇數 (基數)，去掉最後一組 // 這部分可以考慮要不要開
        if (groups.Count % 2 != 0)
        {
            Debug.Log($"Group num is {groups.Count} (odd)，remove last group");
            groups.RemoveAt(groups.Count - 1);
        }*/

        // 如果分組後完全沒有資料，直接回傳空的 data
        if (groups.Count == 0)
        {
            data.notes.Clear();
            return data;
        }

        // 3. 針對 "Group 的數量" 生成隨機 0/1 List
        seed = Random.Range(0, 10000);
        int numGroups = groups.Count;
        Debug.Log($"Group num is {numGroups}");
        int numZeros = numGroups / 2; // 一半的 Group 為 0
        List<int> binaryList = GenerateGroupedBinaryList(numGroups, Config.trial_train_interval, seed);// GenerateRandomBinaryList(numGroups, numZeros, seed);

        // 4. 建立一個新的 List 來存放處理後的 Note
        // 因為有些 Note 可能在分組過程中被丟棄 (不足 n 且時間短)，或者因奇數組被移除
        // 所以我們不能直接修改原有的 data.notes，而是要重組一個新的
        List<OldNote> finalNotes = new List<OldNote>();

        // 5. 遍歷每一個 Group，根據 binaryList 設定顏色與位置
        for (int i = 0; i < groups.Count; i++)
        {
            int typeForGroup = binaryList[i]; // 這一整組要是 0 還是 1
            List<OldNote> currentGroup = groups[i];
            OldNote lastNoteInGroup = null;

            // 計算這一組的時間差 (Last - First)
            float groupDuration = 0f;
            if (currentGroup.Count > 0)
            {
                groupDuration = currentGroup[currentGroup.Count - 1].time - currentGroup[0].time;
            }

            if (currentGroup.Count > 0)
            {
                lastNoteInGroup = currentGroup[currentGroup.Count - 1];
                finalNote.Add(lastNoteInGroup); // 把最後一個 note 標註
            }
            foreach (var note in currentGroup)
            {
                if (typeForGroup == 0) // right blue
                {
                    note.lineIndex = 2;
                    note.type = 1;
                }
                else // left red
                {
                    note.lineIndex = 1;
                    note.type = 0;
                }
                // 將這個 Note 與它所屬 Group 的時間差存入字典，用於 SpawnLine 查巡 group 時間差
                if (!noteGroupDurationMap.ContainsKey(note))
                {
                    noteGroupDurationMap.Add(note, groupDuration);
                }
                // 將設定好的 note 加入最終清單
                finalNotes.Add(note);
            }
        }

        // 6. 將整理好的 note 塞回 data
        data.notes = finalNotes;

        return data;
    }
    // 根據原始 note 創建 group
    public List<List<OldNote>> GroupNotesByRules(List<OldNote> sourceNotes, int n, float beatThreshold_up, float beatThreshold_down, float beatThreshold_1s)
    {
        List<List<OldNote>> groupedNotes = new List<List<OldNote>>();

        // n = 1 or note == null
        if (sourceNotes == null || sourceNotes.Count == 0) return groupedNotes;
        if (n <= 0) n = 1;

        // --- 條件 1: 如果 n = 1，直接每個 Note 一組回傳---
        if (n == 1)
        {
            foreach (var note in sourceNotes)
            {
                groupedNotes.Add(new List<OldNote> { note });
            }
            return groupedNotes;
        }

        int currentIndex = 0;
        int totalNotes = sourceNotes.Count;

        while (currentIndex < totalNotes)
        {
            List<OldNote> currentGroup = new List<OldNote>();
            OldNote firstNote = sourceNotes[currentIndex];
            currentGroup.Add(firstNote);

            // --- 嘗試湊滿 n 個，同時檢查 5 秒限制 ---
            int lookAheadCount = 1;
            while (lookAheadCount < n && (currentIndex + lookAheadCount) < totalNotes)
            {
                OldNote nextNote = sourceNotes[currentIndex + lookAheadCount];
                float timeDiff = nextNote.time - firstNote.time;
                if (timeDiff > beatThreshold_up) break;
                // 5
                // Debug.Log("@@@@@@@@@@@@@@@@@@@@@@@ timeDiff: " + timeDiff + ", beatThreshold_up: "  + beatThreshold_up);
                currentGroup.Add(nextNote);
                lookAheadCount++;
            }

            // --- 條件 3: 檢查是否過短需要丟棄 ---
            bool shouldAddGroup = true;
            if (currentGroup.Count < n)
            {
                float groupDuration = currentGroup[currentGroup.Count - 1].time - currentGroup[0].time;
                // 如果不足 n 且時間差 < 3 秒 -> 丟棄
                if (groupDuration < beatThreshold_down)
                {
                    shouldAddGroup = false;
                }
            }

            // --- 更新索引與應用新規則 ---
            if (shouldAddGroup)
            {
                // 加入有效的 Group
                groupedNotes.Add(currentGroup);

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

        return groupedNotes;
    }
    protected void stop_gmae()
    {
        is_pause = true;
    }

    protected void start_gmae()
    {
        is_pause = false;
    }
}
