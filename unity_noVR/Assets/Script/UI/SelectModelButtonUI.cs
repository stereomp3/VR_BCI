using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.UI;
public class SelectModelButtonUI : MonoBehaviour
{
    public Image targetImage;
    public Image BackImage;
    public TextMeshProUGUI title;
    public TextMeshProUGUI sub_title;
    private SelectModelUI SMU;
    public bool is_first = false;
    // Start is called before the first frame update
    void Start()
    {
        SMU = SelectModelUI.instance;
        if (is_first)
        {
            model_selected();
        }
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    public void model_selected()  // 從 Song_UI_shower.cs 那邊複製 顏色改變設定  Button clickedButton
    {
        if (SMU.last_selected_model == this) return;  // 這邊要設定，如果沒有收到 python 那邊已經完成置換的內容，就卡住
        change_img_color(BackImage, "#CE4760"); // 轉成紅色 // selected
        if (!is_first) change_img_color(SMU.last_selected_model.BackImage, "#1B5174"); // 轉成藍色 // default

        TCP_Client.instance.send_string_to_python(Config.send_python_tcp_select_model_str + Config.separate_str + title.text);

        SMU.last_selected_model = this;
        is_first = false;
    }

    void change_img_color(Image img, string hexColor) // 從 Song_UI_shower.cs 那邊複製
    {
        // 將十六進位顏色轉換成 Color 類型
        if (ColorUtility.TryParseHtmlString(hexColor, out Color newColor))
        {
            // 更新 Image 顏色
            img.color = newColor;
        }
    }

    public void change_title(string txt)
    {
        title.text = txt;
    }

    public void change_sub_title(string txt)
    {
        sub_title.text = txt;
    }
}
