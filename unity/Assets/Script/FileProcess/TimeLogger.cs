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

    private GameManager GM;
    private int trialCount = 0;
    // private string filePath;
    // private Send_LSL_Marker sender;
    private TCP_Client sender;
    void Start()
    {
        // 檔案路徑：儲存在 PersistentDataPath 下，避免平台相容問題
        // filePath = Path.Combine(Application.persistentDataPath, fileName);
        //filePath = fileName; // 會直接生在專案底上
        //File.WriteAllText(filePath, String.Empty); // clear all
        GM = GameManager.instance;
        // sender = Send_LSL_Marker.instance;
        sender = TCP_Client.instance;
        // 開始紀錄協程
        if (GM.is_record_eeg_log) GM.setLogCallback += LogStringAndTime;
    }

    public void LogStringAndTime(int label, LogType type)
    {
        double timestamp = GetUnixTimestamp();

        string log = $"Trial {trialCount} START: {timestamp:F3} LABEL: {label}";
        // Debug.Log(log);
        if (type == LogType.Cut) log = $"Trial {trialCount} CUT: {timestamp:F3}";
        if (type == LogType.End) log = $"Trial {trialCount} END: {timestamp:F3} LABEL: {label}";
        // sender.lsl_send_string_to_python(log) ;
        sender.send_string_to_python(log) ;
        //WriteLineToFile(log);
        if (type == LogType.End) trialCount++;
    }

    double GetUnixTimestamp()
    {
        DateTime now = DateTime.UtcNow;
        DateTime epochStart = new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc);
        return (now - epochStart).TotalSeconds;
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