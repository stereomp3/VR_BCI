using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;

public class ScoreManager : MonoBehaviour
{
    public TextMeshProUGUI score_text;
    public float score = 100; // add score
    private float main_score = 0;
    private GameManager GM;

    // Start is called before the first frame update
    void Start()
    {
        GM = GameManager.instance;
        GM.onSaberCutCallback += add_score;
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    public void add_score()
    {
        main_score += score;
        score_text.text = main_score.ToString();
    }
}
