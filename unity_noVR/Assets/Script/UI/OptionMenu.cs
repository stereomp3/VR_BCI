using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
public class OptionMenu : MonoBehaviour
{
    public Button resumeButton;
    public Button back_Lobby;
    private void Awake()
    {
        resumeButton.onClick.AddListener(ResumeGame);
        back_Lobby.onClick.AddListener(go_to_lobby);
    }

    void ResumeGame()
    {
        GameManager.instance.onGameStartCallback.Invoke();
        gameObject.SetActive(false);    
    }
    void go_to_lobby()
    {
        SceneLoaderManager.instance.LoadLobbyScene();
    }
}
