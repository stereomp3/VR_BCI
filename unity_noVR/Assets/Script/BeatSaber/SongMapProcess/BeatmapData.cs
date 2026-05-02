using Newtonsoft.Json;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

/*[System.Serializable]
public class ColorNote
{
    public float b; // Beat time
    public int x;  //  from 0 (left) to 3 (right)
    public int y; //  from 0 (bottom) to 2 (top)
    public int a; // angle
    public int c; // color (0 = left, 1 = right)
    public int d; // direction
}

[System.Serializable]
public class BeatmapData
{
    public List<ColorNote> colorNotes;
    public bool check_status()
    {
        return false;
    }
}*/

[System.Serializable]
public class OldNote
{
    [JsonProperty("_time")]
    public float time;  // E.g., 6.45, 15.0, 200.32, etc.

    [JsonProperty("_lineIndex")] 
    public int lineIndex;  // 0 (left) to 3 (right)

    [JsonProperty("_lineLayer")]
    public int lineLayer;  // Usually 0 (bottom) to 2 (top)

    [JsonProperty("_type")]
    public int type;  // 0 = left hand (red), 1 = right (blue)

    [JsonProperty("_cutDirection")]
    public int cutDirection;
}

[System.Serializable]
public class OldBeatmap
{
    [JsonProperty("_version")]
    public string version;

    [JsonProperty("_notes")]
    public List<OldNote> notes;
}
