using UnityEngine;
using UnityEngine.UI;
using TMPro; // 使用 TextMeshPro，若使用 UI.Text 則不需要這行
using System.Collections;
using System;
public class DebugUI : MonoBehaviour
{
    // 參考 UI 元件
    public GameObject debugWindow; // 用來控制顯示和隱藏的根物件
    public TextMeshProUGUI debugText; // 用來顯示日誌的 Text 元件
    public ScrollRect scrollRect; // 滾動區域（如果需要）
    public bool autoScroll = true; // 是否自動滾動到底部

    private string logBuffer = ""; // 用來緩存顯示的所有日誌

    void OnEnable()
    {
        // 註冊回調方法
        Application.logMessageReceived += LogCallback;
    }

    void OnDisable()
    {
        // 取消註冊
        Application.logMessageReceived -= LogCallback;
    }
    // 當接收到一條新的日誌訊息時，將其添加到顯示區域
    void LogCallback(string logString, string stackTrace, UnityEngine.LogType type)
    {
        // 你可以根據不同的 logType 添加不同顏色
        string logEntry = $"{System.DateTime.Now:HH:mm:ss} - {logString}\n";

        // 當收到訊息時，將其添加到 logBuffer
        logBuffer += logEntry;

        // 更新顯示的內容
        debugText.text = logBuffer;

        // 自動滾動到底部
        if (autoScroll)
        {
            Canvas.ForceUpdateCanvases();
            scrollRect.verticalNormalizedPosition = 0f; // 滾動到底部
        }
    }

    // 顯示或隱藏 debug 視窗
    public void ToggleDebugWindow()
    {
        debugWindow.SetActive(!debugWindow.activeSelf);
    }

    // 清空所有的 log
    public void ClearLogs()
    {
        logBuffer = "";
        debugText.text = logBuffer;
    }
}
