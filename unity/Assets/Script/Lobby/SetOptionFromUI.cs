using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class SetOptionFromUI : MonoBehaviour
{
    public Scrollbar volumeSlider;
    public TMPro.TMP_Dropdown turnDropdown;
    // public TMPro.TMP_Dropdown forwardDropdown;
    //public SetTurnTypeFromPlayerPref turnTypeFromPlayerPref;

    private void Start()
    {
        volumeSlider.onValueChanged.AddListener(SetGlobalVolume);
        turnDropdown.onValueChanged.AddListener(SetTurnData);
        // forwardDropdown.onValueChanged.AddListener(SetForwardData);

        turnDropdown.SetValueWithoutNotify(GameDataManager.instance.turn_type);
        // forwardDropdown.SetValueWithoutNotify(GameDataManager.instance.forward_type);
    }

    public void SetGlobalVolume(float value)
    {
        AudioListener.volume = value;
    }

    public void SetTurnData(int value)
    {
        GameDataManager.instance.turn_type = value;
    }

    public void SetForwardData(int value)
    {
        GameDataManager.instance.forward_type = value;
    }
}
