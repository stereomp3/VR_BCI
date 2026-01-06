using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public enum Hand // enum 到時候可以統一放在一個地方
{
    lefthand,
    righthand
}

public class EEGTrain : MonoBehaviour  // 用於之前傳統的 MI 
{
    [Header("List Settings")]
    public int totalLength = 60;     // N
    public int numZeros;         // 指定 0 的數量
    public int seed = 42;            // 隨機種子 (可更改控制結果)

    public GameObject cue;
    public GameObject avator;
    private Animator hand_animator;
    
    // fixationTime -> cueTime -> imaginationTime -> intervalTime
    public float fixationTime = 2.0f;     // t=0.00 ~ t=2.00
    public float cueTime = 1.25f;         // t=2.00 ~ t=3.25
    public float imaginationTime = 2.75f; // t=3.25 ~ t=6.00
    public float intervalTime = 1.0f;     // t=6.00 ~ t=7.00

    public List<int> binaryList;

    private int currentTrial = 0;
    private bool isRunning = false;

    GameManager GM;
    void Start()
    {
        GM = GameManager.instance;
        numZeros = totalLength / 2;

        seed = Random.Range(0, 10000);

        binaryList = GenerateRandomBinaryList(totalLength, numZeros, seed);

        // 輸出結果
        Debug.Log("Random Binary List: " + string.Join(", ", binaryList)); // 1 左手 0 右手

        setup_UI_obj();

        // 開始實驗
        StartCoroutine(RunAllTrials());
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    void setup_UI_obj()
    {
        hand_animator = avator.GetComponent<Animator>();
        cue.SetActive(true);
        avator.SetActive(false);   
    }
    List<int> GenerateRandomBinaryList(int N, int numZeros, int seed)
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

    IEnumerator RunAllTrials()
    {
        isRunning = true;
        yield return new WaitForSeconds(3); 
        for (currentTrial = 0; currentTrial < binaryList.Count; currentTrial++)
        {
            int trialType = binaryList[currentTrial]; // 0 or 1 // 1 左手 0 右手
            Hand hand = trialType == 1 ? Hand.lefthand : Hand.righthand;
            Debug.Log($"[Trial {currentTrial + 1}] 類別: {(trialType == 1 ? "左手想像" : "右手想像")}");

            // ===== Phase 1: Fixation + 提示音 =====
            ShowFixation(true);
            PlayBeep();
            yield return new WaitForSeconds(fixationTime);

            if (GM.setLogCallback != null) GM.setLogCallback.Invoke(trialType, LogType.Spawn); // 紀錄 log 到 log.txt

            // ===== Phase 2: 顯示 Cue =====
            ShowCue(hand, true);
            yield return new WaitForSeconds(cueTime);

            // ===== Phase 3: 運動想像 =====
            yield return new WaitForSeconds(imaginationTime);
            ShowCue(hand, false);

            if (GM.setLogCallback != null) GM.setLogCallback.Invoke(trialType, LogType.End);

            // ===== Phase 4: Interval =====
            ShowFixation(false);
            yield return new WaitForSeconds(intervalTime);
        }

        isRunning = false;
        text2speech.instance.Speak("finish");
        Debug.Log("所有 Trial 完成！");
    }

    void ShowFixation(bool show) // 十字改成紅點
    {
        // 這裡可以切換紅點的顯示或隱藏
        if (show)
        {
            cue.SetActive(true);
            Debug.Log("顯示紅點");
        }
        else
        {
            cue.SetActive(false);
            Debug.Log("隱藏紅點");
        }
    }

    void ShowCue(Hand hand, bool show)
    {
        // 顯示左手或右手箭頭
        if (show)
        {
            cue.SetActive(false);
            avator.SetActive(true);
            
            if (hand == Hand.lefthand)
            {
                hand_animator.SetBool("isLeft", true);
                text2speech.instance.Speak("left hand");
            }

            if (hand == Hand.righthand)
            {
                hand_animator.SetBool("isLeft", false);
                text2speech.instance.Speak("right hand");
            }
            Debug.Log($"顯示 Cue: {(hand == Hand.lefthand ? "左手動畫" : "右手動畫")}");
        }
        else
        {
            Debug.Log("隱藏 Cue");
            hand_animator.Rebind();  // 重置到初始狀態
            hand_animator.Update(0f); // 立即刷新到初始姿勢
            avator.SetActive(false);
        }
    }

    void PlayBeep()
    {
        // 播放提示音
        AudioManager.instance.Play("beep");
        Debug.Log("播放提示音");
    }
}
