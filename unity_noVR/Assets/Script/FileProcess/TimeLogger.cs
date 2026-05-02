using UnityEngine;
using System.Collections;
using System.IO;
using System;
using UnityEditor;
public enum LogType
{
    Spawn, // note spawn at specific point // 目前由 forward 發送
    Cut, // cutting note
    End  // undo
}

public class TimeLogger : MonoBehaviour
{

    // public string fileName = "log.txt";
    public static TimeLogger instance;
    private GameManager GM;
    private int trialCount = 0;
    // private string filePath;
    // private Send_LSL_Marker sender;
    private TCP_Client TC;
    private int trial_train_interval_counter;  // 用於 TimeLogger.cs，每 N 個 trial 就更新一次模型，TimeLogger 在 end 的時候，通知更新模型
    #region sigleton
    private void Awake()
    {
        if (instance != null)
        {
            Debug.LogWarning("More than one instance of TimeLogger found!");
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
    int pre_wrong_hit; // 20260227
    AutoSaber AS; 
    void Start()
    {
        // 檔案路徑：儲存在 PersistentDataPath 下，避免平台相容問題
        // filePath = Path.Combine(Application.persistentDataPath, fileName);
        //filePath = fileName; // 會直接生在專案底上
        //File.WriteAllText(filePath, String.Empty); // clear all
        GM = GameManager.instance;
        // sender = Send_LSL_Marker.instance;
        TC = TCP_Client.instance;
        // 開始紀錄協程
        if (GM.is_record_eeg_log) GM.setLogCallback += LogStringAndTime;
        trial_train_interval_counter = Config.trial_train_interval;

        AS = AutoSaber.instance;
        pre_wrong_hit = AS.wrong_hit;
    }

    public void LogStringAndTime(int label, LogType type)
    {
        double timestamp = Timer.GetUnixTimestamp();

        string log = $"Trial {trialCount} START: {timestamp:F3} LABEL: {label}";
        if (type == LogType.Cut) log = $"Trial {trialCount} CUT: {timestamp:F3}";
        if (type == LogType.End) log = $"Trial {trialCount} END: {timestamp:F3} LABEL: {label}";
        // Debug.Log(log);
        // sender.lsl_send_string_to_python(log) ;
        TC.send_string_to_python(log) ;
        //WriteLineToFile(log);
        if (type == LogType.End)
        {
            GM.onSaberCutCallback.Invoke(); // 處裡 UI 計分或是其他相關事件
            trialCount++;
            trial_train_interval_counter -= 1;
            if (trial_train_interval_counter <= 0)
            {
                trial_train_interval_counter = Config.trial_train_interval;
                if (Config.adaptive_model || GameDataManager.instance.is_calibration)//  
                {
                    float score;
                    if (AS.correct_hit + AS.wrong_hit == 0) score = 0;
                    else score = (float)AS.correct_hit / (float)(AS.correct_hit + AS.wrong_hit);

                    if (AS.wrong_hit != pre_wrong_hit && score < 0.75) 
                    {
                        TC.send_string_to_python(Config.send_python_tcp_calibration_start);
                        Debug.Log("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@: correct 4 trial");
                    }
                    else // 分數大於等於 0.75 或是 4 個 trial 全對，就就不更新模型
                    {
                        TC.send_string_to_python("Correct 4 trial or acc > 0.75. Don't update model");
                    }
                    pre_wrong_hit = AS.wrong_hit;
                }
            }
        }
    }

    public void reset_trialCount() // 重新設定傳送的 trial count，用於 Calibration2 (因為它會送給 python 那邊停止 log 並重寫的功能)
    {
        trialCount = 0; 
    }
    /*void WriteLineToFile(string line)
    {
        try
        {
            File.AppendAllText(filePath, line + Environment.NewLine);
        }
        catch (Exception e)
        {
            Debug.LogError("寫入檔案錯誤: " + e.Message);
        }
    }*/
}