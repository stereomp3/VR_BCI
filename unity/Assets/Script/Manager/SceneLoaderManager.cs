using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
public class SceneLoaderManager : MonoBehaviour
{
    #region Singleton 
    public static SceneLoaderManager instance;
    void Awake()
    {
        if (instance != null)
        {
            Debug.LogWarning("More than one instance of SceneManager found!");
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

    // Start is called before the first frame update
    public Animator transition;
    public float transitionTime = 1f;
    public void LoadLobbyScene()
    {
        GameDataManager.instance.is_calibration = false;
        StartCoroutine(LoadScene(Config.Stage[(int)Stage.LOBBY], 0));
    }
    public void LoadMIStage()
    {
        StartCoroutine(LoadScene(Config.Stage[(int)Stage.MI], 1));
    }
    public void LoadBeatSaberStage()
    {
        StartCoroutine(LoadScene(Config.Stage[(int)Stage.BEATSABER], 1));
    }
    public void LoadCalibrationStage()
    {
        StartCoroutine(LoadScene(Config.Stage[(int)Stage.CALIBRATION], 1));
    }
    /*public void ClickSound()
    {
        AudioManager.instance.Play("Click");
    }*/

    public void Exit()
    {
        Application.Quit();
    }
    IEnumerator LoadScene(string scene, int index)
    {
        //AudioManager.instance.Play("Click");
        if (GameManager.instance.is_training)
        {
            yield return null;
        }
        else
        {
            transition.SetTrigger("Start");
            yield return new WaitForSeconds(transitionTime);

            // Send_LSL_Marker.instance.lsl_send_string_to_python(scene);
            // Send_LSL_Marker.instance.CloseStream();
            TCP_Client.instance.send_string_to_python(scene);
            SceneManager.LoadScene(scene);
            // GameManager.instance.SceneIndex = index;
        }
    }
}
