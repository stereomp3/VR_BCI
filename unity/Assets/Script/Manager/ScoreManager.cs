using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;

public class ScoreManager : MonoBehaviour
{
    public TextMeshProUGUI score_text;
    // public float score = 100; // add score
    // private float main_score = 0;
    private GameManager GM;
    private AutoSaber AS;
    // Start is called before the first frame update
    void Start()
    {
        GM = GameManager.instance;
        GM.onSaberCutCallback += show_score;
        AS = AutoSaber.instance;    
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    public void show_score()
    {
        // main_score += score;
        string rank;
        float score;
        if (AS.correct_hit + AS.wrong_hit == 0) score = 0;
        else score = (float)AS.correct_hit / (float)(AS.correct_hit + AS.wrong_hit);


        if (score > 0.9f) rank = "S";
        else if (score > 0.8f) rank = "A";
        else if (score > 0.7f) rank = "B";
        else if (score > 0.6f) rank = "C";
        else if (score > 0.5f) rank = "D";
        else rank = "E";
        score_text.text = $"{rank} :  {AS.correct_hit:D2} / {(AS.correct_hit + AS.wrong_hit):D2}";
    }
}
