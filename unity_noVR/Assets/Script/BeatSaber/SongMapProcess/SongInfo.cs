using System.Collections.Generic;

[System.Serializable]
public class SongInfo
{
    public string _songName;
    public string _songAuthorName;
    public string _coverImageFilename;
    public string _songFilename;

    public float _beatsPerMinute;
    public float _songTimeOffset;
    public float _previewStartTime;
    public float _previewDuration;

    public List<DifficultyBeatmapSet> _difficultyBeatmapSets;
}

[System.Serializable]
public class DifficultyBeatmapSet
{
    public string _beatmapCharacteristicName;
    public List<DifficultyBeatmap> _difficultyBeatmaps;
}

[System.Serializable]
public class DifficultyBeatmap
{
    public string _difficulty;
    public int _difficultyRank;
    public string _beatmapFilename;
    public float _noteJumpMovementSpeed;
    public float _noteJumpStartBeatOffset;
}