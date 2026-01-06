using System;
using System.Collections;
using System.Collections.Generic;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;
using static Oculus.Interaction.Context;


public class TCP_Client : MonoBehaviour
{
    public static TCP_Client instance;
    TcpClient client;
    NetworkStream stream;

    private byte[] buffer = new byte[1024];

    // record the 10 (or more in furture) history of lsl (this is for other script to read)
    public Queue<int> valueHistory = new Queue<int>();
    public int historyLength = 10;
    // for simulate input
    public bool is_simulated = true;
    private float updateInterval = 0.1f; // LSL or input interval
    private float timer = 0f;
    public int simulatedInput = 0; // 0 or 1 
    private GameManager GM;
    Thread receiveThread;

    // 用來存放要在主執行緒執行的任務
    private readonly Queue<Action> mainThreadActions = new Queue<Action>();

    [Header("LSL sender parameter")]
    public bool send_info_to_python = false;
    #region sigleton
    private void Awake()
    {
        historyLength = Config.PredictionHistoryLength;
        /*client = new TcpClient("127.0.0.1", 50007);
        stream = client.GetStream();*/
        if (instance != null)
        {
            Debug.LogWarning("More than one instance of GameManager found!");
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
    void Start()
    {
        GM = GameManager.instance;
        StartCoroutine(ConnectWithTimeout(Config.TCP_HOST, Config.TCP_PORT, 2f)); // 2 秒超時
            
    }

    // Update is called once per frame
    void Update()
    {
        if (stream == null && is_simulated)
        {
            timer += Time.deltaTime;
            if (timer >= updateInterval)
            {
                timer -= updateInterval;
                update_history(simulatedInput);
            }
        }
        else
        {
            if (send_info_to_python)
            {
                send_string_to_python("test");
                send_info_to_python = false;
            }

            // 每一幀在主執行緒執行佇列中的任務
            lock (mainThreadActions)
            {
                while (mainThreadActions.Count > 0)
                    mainThreadActions.Dequeue().Invoke();
            }
        }
    }
    void ReceiveLoop()
    {
        byte[] buffer = new byte[1024];
        send_string_to_python(Config.send_python_tcp_model_str); // send
        while (true)
        {
            int bytes = stream.Read(buffer, 0, buffer.Length);
            string marker = Encoding.UTF8.GetString(buffer, 0, bytes);
            if (marker == "0" || marker == "1") update_history(int.Parse(marker));  // if mark is prediction
            Debug.Log("Received from Python: " + marker);

            if (marker.Contains(Config.receive_python_tcp_model_str)) // 需要為 Config.receive_python_tcp_model_str 的內容才會觸發
            {
                mainThreadActions.Enqueue(() => set_python_models_name(marker)); 
            }

            if (GM.is_training && (marker != "0" && marker != "1")) // GM.is_training 在 Calibration 後面觸發或是 按下 training button 觸發
            {
                lock (mainThreadActions) // 將需要在主執行緒執行的邏輯排入佇列，因為 thread 不能呼叫 UI 功能
                {
                    // 這裡的程式會在主執行緒中執行
                    mainThreadActions.Enqueue(() => GM.onTrainStartCallback.Invoke(marker));
                }
            }
        }
    }

    void set_python_models_name(string inputString)
    {
        // 找到 Config.receive_python_tcp_model_str 之後的字串
        int startIndex = inputString.IndexOf(Config.receive_python_tcp_model_str) + Config.receive_python_tcp_model_str.Length;

        // 檢查是否能找到 Config.receive_python_tcp_model_str
        if (startIndex > 0)
        {
            string modelsPart = inputString.Substring(startIndex);

            // 以 Config.separate_str 分隔字串
            string[] models = modelsPart.Split(new string[] { Config.separate_str }, StringSplitOptions.None);
            List<string> modelList = new List<string>(models);
            modelList.RemoveAt(0); // 移除第一個空值
            // 把分割後的字串加入 List
            GameDataManager.instance.python_models_name = modelList;
            if(GM.onOnPythonModelSetCallback != null)
                GM.onOnPythonModelSetCallback.Invoke();
        }
    }

    public void send_string_to_python(string message)
    {
        if (stream != null)
        {
            byte[] data = Encoding.UTF8.GetBytes(message);
            stream.Write(data, 0, data.Length);
        }
    }
    public void update_history(int value) // update history queue
    {
        // Add new value and keep the queue size
        valueHistory.Enqueue(value);
        if (valueHistory.Count > historyLength)
            valueHistory.Dequeue();
    }
    void OnApplicationQuit()
    {
        stream.Close();
        client.Close();
    }

    IEnumerator ConnectWithTimeout(string host, int port, float timeoutSeconds)
    {
        client = new TcpClient();
        bool isConnected = false;

        // 在背景執行緒中嘗試連線
        Thread connectThread = new Thread(() =>
        {
            try
            {
                client.Connect(host, port);
                isConnected = true;
            }
            catch (Exception e)
            {
                Debug.LogError($"❌ 連線錯誤: {e.Message}");
            }
        });

        connectThread.Start();

        // 等待 timeout 時間
        float timer = 0f;
        while (!isConnected && timer < timeoutSeconds)
        {
            timer += Time.deltaTime;
            yield return null;
        }

        if (isConnected && client.Connected)
        {
            Debug.Log("✅ 連線成功！");
            stream = client.GetStream();
            // stream.ReadTimeout = 5000;
            // stream.WriteTimeout = 5000;

            receiveThread = new Thread(ReceiveLoop);
            receiveThread.IsBackground = true;  // 像是 python 的 daemon=True，主程序中斷，就跟者斷掉
            receiveThread.Start();
            is_simulated = false;
            GM.use_LSL_to_controll_saber = true;
            
        }
        else
        {
            Debug.LogWarning("⏰ 連線逾時（超過 2 秒）！");
            is_simulated = true;
            GM.use_LSL_to_controll_saber = false;
            try { client.Close(); } catch { }
        }

        // 確保連線執行緒結束
        if (connectThread.IsAlive)
            connectThread.Abort();
    }
}
