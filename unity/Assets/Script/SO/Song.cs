using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;


[CreateAssetMenu(fileName = "New Song", menuName = "SO/Song")]
public class Song : ScriptableObject
{
    public new string name;  // 讓其他Scriptable也可以用name
    public string description;

    public string info_name = "info.dat"; // info 名稱

    public string dir;  // 歌曲資料夾位置

    public string[] file_names;  // 對應各種難度和模式的文件名稱
    public string[] ui_show_names;  // 顯示在 UI 上面的名稱，需要對應 file_name 數量

    void update_song(Song song) // 用於更新 clone 的 song
    {
        song.name = name;
        song.description = description;
        song.info_name = info_name;
        song.dir = dir;
        song.file_names = file_names;
        song.ui_show_names = ui_show_names;
    }
    // 把 json 更新成目標的 song，用在 selected_song，在 Song_UI_shower.cs 觸發，存成 json
    public void SaveSettings()
    {
        string json = JsonUtility.ToJson(this);
        string fullPath = Path.Combine(Application.persistentDataPath, Config.song_json_file);
        File.WriteAllText(fullPath, json);
    }

    public void LoadSettings()
    {
        string levelPath = Path.Combine(Application.persistentDataPath, Config.level_json_file);
        string songPath = Path.Combine(Application.persistentDataPath, Config.song_json_file);
        
        if (File.Exists(songPath))
        {
            if (File.Exists(levelPath))
            {
                string levelDataJson = File.ReadAllText(levelPath);
                string SongJson = File.ReadAllText(songPath);
                JsonUtility.FromJsonOverwrite(SongJson, this);
                Song filteredSong = FilterSongByLevelData(JsonUtility.FromJson<LevelData>(levelDataJson), this);
                string json = JsonUtility.ToJson(filteredSong);
                JsonUtility.FromJsonOverwrite(json, this);
            }
            else // only use the song json, select first MI
            {
                string json = File.ReadAllText(songPath);
                JsonUtility.FromJsonOverwrite(json, this);
            }
        }
    }

    // 用來從 json 讀取 LevelData 和 Song，並過濾出對應的資料
    public Song FilterSongByLevelData(LevelData levelData, Song song)
    {
        // 根據 LevelData 中的 level 進行過濾
        FilterByName(levelData.level, song);

        return song;
    }
    public void FilterByName(string name, Song song) // 用來把歌曲變成一個
    {
        // 找到匹配的索引
        int index = System.Array.IndexOf(song.ui_show_names, name);

        if (index != -1) // 如果找到了匹配的名稱
        {
            // 保留對應索引的值，並清空其他元素
            song.ui_show_names = new string[] { song.ui_show_names[index] };
            song.file_names = new string[] { song.file_names[index] };
        }
        else
        {
            // 如果沒有找到，給予錯誤提示
            Debug.LogWarning("No match found for the name: " + name);
        }
    }
    // 更新 ui_show_names 的方法
    public void UpdateUIShowNames()
    {
        ui_show_names = new string[file_names.Length]; // 确保 ui_show_names 有正确长度

        for (int i = 0; i < file_names.Length; i++)
        {
            string fileName = file_names[i];
            ui_show_names[i] = "Unknown";
            
            foreach (var item in Config.level_ui_strings)
            {
                if (fileName.Contains(item))
                {
                    ui_show_names[i] = item;
                    break;
                }
            }
            if (fileName.Contains("ExpertPlus"))
            {
                ui_show_names[i] = "Expert+";
            }
            if (fileName.Contains("MI"))
            {
                ui_show_names[i] = "MI";
            }
        }
    }
}
