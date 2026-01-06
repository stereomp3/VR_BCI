#if UNITY_EDITOR
using UnityEditor;
using UnityEngine;

[CustomEditor(typeof(Song))]
public class SongEditor : Editor
{
    public override void OnInspectorGUI()
    {
      
        Song song = (Song)target;

        // 顯示原本的 Inspector UI
        DrawDefaultInspector();

        // 檢查 file_names 是否有變化
        if (song.file_names != null && song.file_names.Length > 0)
        {
            // 如果 file_names 内容有变化，更新 ui_show_names
            song.UpdateUIShowNames();
        }

        // 檢查 file_names 是否有變化，重新繪製
        if (GUI.changed)
        {
            EditorUtility.SetDirty(song); // 告诉 Unity 數據更改了
        }
    }
}
#endif