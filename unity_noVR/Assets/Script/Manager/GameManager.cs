using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;

public class GameManager : MonoBehaviour
{
    public static GameManager instance;
    #region sigleton
    private void Awake()
    {
        if (instance != null)
        {
            Debug.LogWarning("More than one instance of GameManager found!");
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

    [Header("Saber setting")]
    public bool is_audosaber = false;
    public bool use_LSL_to_controll_saber = false;  // 在 TCP_Client(old: ReceiveLSLMarker.cs) 裡面如果 有連線到 LSL，這個會設定為 true，反之為 false，在 AutoSaber.cs 裡面也有用到
    [Header("record setting")]
    public bool is_record_eeg_log = false;
    [Header("Hand setting")]
    public float min_show_hand_distance = 0.1f;

    [Header("final UI")]
    public GameObject finalCanvus;
    public TextMeshProUGUI finalAcc;
    [Header("final UI")]
    public GameObject StopOptionCanvas;
    
    // event setting
    public delegate void OnNoteSpawn(); 
    public delegate IEnumerator OnAutoSaber(int note_x, int note_y, int note_direction, bool use_brain_to_controll); 
    public delegate void OnSaberCut(); 
    public delegate void SetLog(int label, LogType type);
    public delegate void OnGameStop();
    public delegate void OnGameStart();
    public delegate void OnTrainStart(string marker);
    public delegate void OnPythonModelSet();
    public OnNoteSpawn onNoteSpawnCallback; // 0 event
    public OnAutoSaber onAutoSaberCallback;  // 1 event in AutoSaber，trigger in BeatmapSpawner.cs，處裡 autosaber 位置移動
    public OnSaberCut onSaberCutCallback;  // 1 event in ScoreManager，1 event in CalibrationBeatmapSpawner，trigger in TimeLogger.cs，處裡切的時候相關事件
    public SetLog setLogCallback;  // 1 event in TimeLogger，trigger in NoteLogTrigger and SaberSlicer and EEGTrain，紀錄 Log
    public OnGameStop onGameStopCallback;  // 用在 BeatSaberInforLoader.cs 停歌，和 BeatmapSpawener 停送 note，在 GameManager 裡面雙手合起來的時候呼叫
    public OnGameStart onGameStartCallback;  // 用在 BeatSaberInforLoader.cs 和 BeatmapSpawener 停送 note，在 OptionMenu 設定 Canvas button 觸發的時候呼叫
    public OnTrainStart onTrainStartCallback;  // 用在 TrainingLogUI.cs ，在 TCP_Client 接收到 config.training_start 觸發的時候呼叫
    public OnPythonModelSet onOnPythonModelSetCallback;  // 用在 SelectModelUI.cs ，在 TCP_Client 接收到 Config.receive_python_tcp_model_str，並設定完成 GameDataManager 裡面的 python_models_name 的時候觸發的時候呼叫
    // event setting end0
    // UI event setting 
    //
    // UI event setting end0
    [Header("Other setting")]
    public int correct_slice = 0; // tmp;
    public int correct_total = 0; // tmp;
    public float song_delay_time = 2.5f; // // 會事先產生 cube，然後到定點時間出現在指定位置，use in BeatmapSpawner, BeatSaberInfoLoader, forward
    public float start_game_seconds = 0f;  // 遊戲開始時間，也是 calibration 的時間  // use in BeatmapSpawner, BeatSaberInfoLoader
    public Stage now_stage;  // 紀錄目前在的位置
    public bool is_training = false; // 紀錄 python 那邊是否在訓練，透過 LSL Train_MarkerStream stream 傳送的 string，來設定是否有訓練，用於 SceneLoaderManager LoadScene
    public bool calibration_model_finish =false; // 透過 TCP_Client 改變，這個為 true，代表 calibration 的模型訓練完成，然後由 Calibration2.cs 來變為 false (waiting_loop)

    private Transform LeftHand_parent, RightHand_parent;
    private bool is_show_hand = true;
    private float detect_hand_time = 1f;
    private float timer;
    private bool isPaused = false; // 用於遊戲暫停
    
    
    // Start is called before the first frame update
    void Start()
    {
        Application.runInBackground = true;
        // Send_LSL_Marker.instance.lsl_send_string_to_python(Config.Stage[(int)now_stage]);
        TCP_Client.instance.send_string_to_python(Config.Stage[(int)now_stage]);

        if (StopOptionCanvas != null) StopOptionCanvas.SetActive(false);

        if (is_audosaber)
        {
            // onGameStartCallback += set_hands_unactive;
        }
        if (!is_audosaber && AutoSaber.instance != null) AutoSaber.instance.gameObject.SetActive(false);
    }

    // Update is called once per frame
    void Update()
    {
        if(Input.GetKeyDown(KeyCode.Escape))
        {
            TogglePause();
        }
        /*timer += Time.deltaTime; // 暫時把暫停功能關閉，受試者很多會不小心用到
        if (timer > detect_hand_time)
        {
            // visible the hand or unvisible the hand
            if ((LeftHand_parent.position - RightHand_parent.position).magnitude < min_show_hand_distance && !is_training)
            {
                set_hands_active();
                if (now_stage != Stage.LOBBY) TogglePause();
            }
            timer = 0;
        }*/
    }


    public IEnumerator SetFinalAcc() // tmp // 之後要改成歌曲完成後出現 // 目前在 BeatmapSpawner 呼叫
    {
        yield return new WaitForSeconds(5f);
        AutoSaber AS = AutoSaber.instance;
        if (AS != null && is_audosaber)
        {
            finalCanvus.SetActive(true);    
            finalAcc.text = AS.correct_hit.ToString() + " / " + (AS.correct_hit + AS.wrong_hit).ToString() + " = " + ((float)AS.correct_hit / (float)(AS.correct_hit + AS.wrong_hit)).ToString();
            TCP_Client.instance.send_string_to_python(finalAcc.text);
        }
        else
        {
            finalCanvus.SetActive(true);
            finalAcc.text = correct_slice.ToString() + " / " + (correct_total).ToString() + " = " + ((float)correct_slice / (float)correct_total).ToString();
            TCP_Client.instance.send_string_to_python(finalAcc.text);
        }
    }

    // 切換暫停狀態
    void TogglePause()
    {
        if (StopOptionCanvas != null)
        {
            StopOptionCanvas.SetActive(true);  // 顯示暫停選項
        }
        // 暫停遊戲
        onGameStopCallback.Invoke();
    }


    public void set_autosaber() // 用於 calibration 把手變不見，然後把 auto saber 叫出來
    {
        is_audosaber = true;
    }

    public void set_un_autosaber()
    {
        is_audosaber = false;
        if (!is_audosaber && AutoSaber.instance != null) AutoSaber.instance.gameObject.SetActive(false);
    }

    public void start_training() // 開始 training 或是 calibration
    {
        // ReceiveTrainLSLMarker.instance.set_lsl_stream();
        // Send_LSL_Marker.instance.lsl_send_string_to_python(Config.Stage[(int)Stage.TRAIN]); // go in training state
        TCP_Client.instance.send_string_to_python(Config.Stage[(int)Stage.TRAIN]);
        is_training = true;
    }
}
