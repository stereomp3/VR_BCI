using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
public class TutorialContinueUI : MonoBehaviour
{
    public Button pre_button;
    public Button nxt_button;
    // Reference to the parent GameObject
    private GameObject parentObject;

    // References to the previous and next GameObjects
    public GameObject previousObject;
    public GameObject nextObject;
    // Start is called before the first frame update
    void Start()
    {
        if (pre_button!= null) pre_button.onClick.AddListener(set_previousObject_active);
        if (nxt_button!= null) nxt_button.onClick.AddListener(set_nextObject_active);

        parentObject = transform.parent.gameObject;

        // Check if the parent object is assigned
        if (parentObject == null)
        {
            Debug.LogError("Parent object is not assigned.");
            return;
        }

        // Get all the children of the parent object
        var childList = new List<Transform>();
        for (int i = 0; i < parentObject.transform.childCount; i++)
        {
            childList.Add(parentObject.transform.GetChild(i));
        }

        childList.Remove(parentObject.transform);

        // Find this GameObject in the list of children
        int currentIndex = childList.FindIndex(child => child.gameObject == gameObject);

        if (currentIndex != -1)
        {
            // Set the previous and next objects
            previousObject = currentIndex > 0 ? childList[currentIndex - 1].gameObject : null;
            nextObject = currentIndex < childList.Count - 1 ? childList[currentIndex + 1].gameObject : null;

            // Log the result for debugging purposes
            Debug.Log($"{gameObject.name}: Previous = {(previousObject != null ? previousObject.name : "null")}, Next = {(nextObject != null ? nextObject.name : "null")}");
        }
        else
        {
            Debug.LogError("Current GameObject is not a child of the parent object.");
        }
    }


    public void set_previousObject_active()
    {
        if (previousObject != null) previousObject.SetActive(true);
        else return;

        gameObject.SetActive(false);
    }

    public void set_nextObject_active()
    {
        if (nextObject != null) nextObject.SetActive(true);
        else return;
        gameObject.SetActive(false);
    }
}
