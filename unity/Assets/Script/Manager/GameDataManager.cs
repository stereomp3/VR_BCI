using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GameDataManager : MonoBehaviour
{
    // record option data, setting ...
    #region Singleton 
    public static GameDataManager instance;
    void Awake()
    {
        if (instance != null)
        {
            Debug.LogWarning("More than one instance of GameDataManager found!");
            Destroy(gameObject);
            return;
        }
        else
        {
            DontDestroyOnLoad(gameObject);
            instance = this;
        }
    }

    #endregion

    public List<string> python_models_name = new List<string>();
    public int turn_type = 0;  // { Snap = 0, Continuous = 1};
    public int forward_type = 0;  // { follow camera = 0, follow palyer = 1};
}
