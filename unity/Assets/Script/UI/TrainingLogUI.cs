using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.UI;
public class TrainingLogUI : MonoBehaviour
{
    [Header("UI setting")]
    public TextMeshProUGUI Training_log_txt;
    public GameObject Training_UI;
    public Button back_Lobby;

    public int maxLines = 4;
    private List<string> lines = new List<string>();

    private string[] sample;
    private const int channelCount = 1;

    private GameManager GM;


    private void Start()
    {
        GM = GameManager.instance;
        GM.onTrainStartCallback += AddMarker;
    }

    // 將新的 marker 加入到文字中
    void AddMarker(string marker)
    {
        Training_UI.SetActive(true);
        // Debug.Log("WWWWWWWWWWWWWWWWWWWWWWWWW marker: " + marker);
        if (marker == Config.training_done)
        {
            GM.is_training = false;
            back_Lobby.gameObject.SetActive(true);
            back_Lobby.onClick.AddListener(SceneLoaderManager.instance.LoadLobbyScene);
        }
        // 如果行數超過了最大值（10行），則移除第一行
        if (lines.Count >= maxLines)
        {
            lines.RemoveAt(0); // 移除第一行
        }

        // 新增新的 marker 到最後一行
        lines.Add(marker);

        // 更新 TextMeshProUGUI 的顯示內容
        Training_log_txt.text = string.Join("\n", lines);
    }
}
