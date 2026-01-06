using UnityEngine;

public static class Config
{
    // json
    public static string level_json_file = "levelData.json";
    public static string song_json_file = "song.json";
    public static string[] level_ui_strings = { "MI", "Easy", "Normal", "Hard", "Expert", "ExpertPlus" }; // 對應下面的 enum Level
    public static string Calibration = "Calibration"; // 紀錄 Calibration 資料夾的名稱，會用於去 Calibration 場景的字串比對

    // TCP setting (過去為 lsl 設定
    public static string receive_python_lsl_stream = "MarkerStream";
    public static string receive_python_train_lsl_stream = "Train_MarkerStream";
    public static string to_python_lsl_stream = "UnityMarkerStream";
    public static string training_done = "training done";  // python 傳送開始訓練好的字串內容，用於 TCP_Clients (old ReceiveLSLTrainMarker.cs) // 在 TrainingLogUI.cs AddMarker function 裡面
    public static int TCP_PORT = 50007;
    public static string TCP_HOST = "127.0.0.1"; // 172.20.2.173
    public static int PredictionHistoryLength = 100;
    // stage 對應 enum Stage 順序 
    public static string[] Stage = { "Lobby" , "EEG_Calibration", "MI" , "BeatSaber" , "Training" };

    // 模型選擇設定 in TCP_Client.cs
    public static string send_python_tcp_model_str = "send_python_tcp_model_str"; // 在 lobby 就會送出這個字串 ()，然後 python 會回傳下面字串和 model 名稱
    public static string receive_python_tcp_model_str = "SENT_UNITY_MODEL_STR"; // receive from python txt
    public static string send_python_tcp_select_model_str = "send_python_tcp_select_model_str"; // 在 SelectModelButtonUI.cs 裡面，function 呼叫
    public static string separate_str = "@@@"; // 用於分開字串的符號，python unity 都使用這個

    // audio
    public static float volume = 0.3f;

    // other
    public static bool pass_tutorial = false; // pass_tutorial 主要跳過教學，然後 calibration 前面的 ME 也變成 MI，算是快速模式，用於測試

    // 遊戲設定
    public static int group_note_num = 1; // 設定一組 group 裡面有幾個 note
    public static float cube_space_time = 0.7f; // 對應地圖的產出間隔
}

public enum Stage // Config.Stage[(int)Stage.LOBBY] 對應 Stage 字串
{
    LOBBY,
    CALIBRATION,
    MI,
    BEATSABER,
    TRAIN
}

public enum Level
{
    MI,
    Easy,
    Normal,
    Hard,
    Expert,
    ExpertPlus
}