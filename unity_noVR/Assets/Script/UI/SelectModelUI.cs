using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
public class SelectModelUI : MonoBehaviour
{
    public static SelectModelUI instance;
    #region sigleton
    private void Awake()
    {
        if (instance != null)
        {
            Debug.LogWarning("More than one instance of SelectModelUI found!");
            Destroy(gameObject);
            return;
        }
        else
        {
            // DontDestroyOnLoad(gameObject);
            instance = this;
        }
    }
    #endregion

    public GameObject model_content;
    public GameObject buttonUI_prefab;
    public SelectModelButtonUI last_selected_model;
    // Start is called before the first frame update
    void Start()
    {
        GameManager.instance.onOnPythonModelSetCallback += set_model_UI;
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    void set_model_UI()
    {
        foreach (string name in GameDataManager.instance.python_models_name)
        {
            SelectModelButtonUI SMBU = Instantiate(buttonUI_prefab, model_content.transform).GetComponent<SelectModelButtonUI>();
            SMBU.change_title(name);
            SMBU.change_sub_title("trained_model");
        }
    }
}
