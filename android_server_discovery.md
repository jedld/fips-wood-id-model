# Android UDP Server Discovery

This document provides Android code to automatically detect the Wood ID Upload Server via UDP broadcast.

## Required Permissions

Add these permissions to your `AndroidManifest.xml`:

```xml
<uses-permission android:name="android.permission.INTERNET" />
<uses-permission android:name="android.permission.ACCESS_NETWORK_STATE" />
<uses-permission android:name="android.permission.ACCESS_WIFI_STATE" />
<uses-permission android:name="android.permission.CHANGE_WIFI_MULTICAST_STATE" />
```

## Kotlin Implementation

### 1. Server Discovery Service

```kotlin
import android.app.Service
import android.content.Intent
import android.os.Binder
import android.os.IBinder
import android.util.Log
import kotlinx.coroutines.*
import org.json.JSONObject
import java.net.*

class ServerDiscoveryService : Service() {
    private val TAG = "ServerDiscovery"
    private val BROADCAST_PORT = 5001
    private val DISCOVERY_TIMEOUT = 10000L // 10 seconds
    
    private val binder = LocalBinder()
    private var discoveryJob: Job? = null
    private var discoveredServers = mutableListOf<ServerInfo>()
    
    inner class LocalBinder : Binder() {
        fun getService(): ServerDiscoveryService = this@ServerDiscoveryService
    }
    
    override fun onBind(intent: Intent): IBinder {
        return binder
    }
    
    data class ServerInfo(
        val type: String,
        val ip: String,
        val port: Int,
        val name: String,
        val version: String
    )
    
    fun startDiscovery(onServerFound: (ServerInfo) -> Unit) {
        stopDiscovery()
        
        discoveryJob = CoroutineScope(Dispatchers.IO).launch {
            try {
                val socket = DatagramSocket(BROADCAST_PORT)
                socket.broadcast = true
                socket.soTimeout = DISCOVERY_TIMEOUT.toInt()
                
                Log.d(TAG, "Listening for server broadcasts on port $BROADCAST_PORT")
                
                val buffer = ByteArray(1024)
                val packet = DatagramPacket(buffer, buffer.size)
                
                while (isActive) {
                    try {
                        socket.receive(packet)
                        val message = String(packet.data, 0, packet.length)
                        
                        val serverInfo = parseServerInfo(message)
                        if (serverInfo != null && serverInfo.type == "wood_id_server") {
                            Log.d(TAG, "Server found: ${serverInfo.name} at ${serverInfo.ip}:${serverInfo.port}")
                            
                            // Check if server is not already in the list
                            if (!discoveredServers.any { it.ip == serverInfo.ip }) {
                                discoveredServers.add(serverInfo)
                                onServerFound(serverInfo)
                            }
                        }
                    } catch (e: SocketTimeoutException) {
                        // Timeout is expected, continue listening
                        continue
                    } catch (e: Exception) {
                        Log.e(TAG, "Error receiving broadcast: ${e.message}")
                    }
                }
                
                socket.close()
            } catch (e: Exception) {
                Log.e(TAG, "Discovery error: ${e.message}")
            }
        }
    }
    
    fun stopDiscovery() {
        discoveryJob?.cancel()
        discoveryJob = null
    }
    
    fun getDiscoveredServers(): List<ServerInfo> {
        return discoveredServers.toList()
    }
    
    private fun parseServerInfo(message: String): ServerInfo? {
        return try {
            val json = JSONObject(message)
            ServerInfo(
                type = json.getString("type"),
                ip = json.getString("ip"),
                port = json.getInt("port"),
                name = json.getString("name"),
                version = json.getString("version")
            )
        } catch (e: Exception) {
            Log.e(TAG, "Error parsing server info: ${e.message}")
            null
        }
    }
    
    override fun onDestroy() {
        stopDiscovery()
        super.onDestroy()
    }
}
```

### 2. Main Activity Example

```kotlin
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.os.Bundle
import android.os.IBinder
import android.util.Log
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView

class MainActivity : AppCompatActivity() {
    private val TAG = "MainActivity"
    
    private lateinit var statusText: TextView
    private lateinit var startDiscoveryButton: Button
    private lateinit var serversRecyclerView: RecyclerView
    private lateinit var serversAdapter: ServersAdapter
    
    private var discoveryService: ServerDiscoveryService? = null
    private var bound = false
    
    private val connection = object : ServiceConnection {
        override fun onServiceConnected(className: ComponentName, service: IBinder) {
            val binder = service as ServerDiscoveryService.LocalBinder
            discoveryService = binder.getService()
            bound = true
            Log.d(TAG, "Service connected")
        }
        
        override fun onServiceDisconnected(arg0: ComponentName) {
            bound = false
            Log.d(TAG, "Service disconnected")
        }
    }
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        statusText = findViewById(R.id.statusText)
        startDiscoveryButton = findViewById(R.id.startDiscoveryButton)
        serversRecyclerView = findViewById(R.id.serversRecyclerView)
        
        serversAdapter = ServersAdapter { serverInfo ->
            // Handle server selection
            connectToServer(serverInfo)
        }
        
        serversRecyclerView.layoutManager = LinearLayoutManager(this)
        serversRecyclerView.adapter = serversAdapter
        
        startDiscoveryButton.setOnClickListener {
            if (bound) {
                startServerDiscovery()
            }
        }
        
        // Bind to the service
        Intent(this, ServerDiscoveryService::class.java).also { intent ->
            bindService(intent, connection, Context.BIND_AUTO_CREATE)
        }
    }
    
    private fun startServerDiscovery() {
        statusText.text = "🔍 Searching for servers..."
        startDiscoveryButton.isEnabled = false
        
        discoveryService?.startDiscovery { serverInfo ->
            runOnUiThread {
                statusText.text = "✅ Server found: ${serverInfo.name}"
                serversAdapter.addServer(serverInfo)
                startDiscoveryButton.isEnabled = true
            }
        }
    }
    
    private fun connectToServer(serverInfo: ServerDiscoveryService.ServerInfo) {
        // Here you would implement the connection logic
        // For example, open a new activity to upload images
        Log.d(TAG, "Connecting to server: ${serverInfo.ip}:${serverInfo.port}")
        
        val intent = Intent(this, UploadActivity::class.java).apply {
            putExtra("server_ip", serverInfo.ip)
            putExtra("server_port", serverInfo.port)
            putExtra("server_name", serverInfo.name)
        }
        startActivity(intent)
    }
    
    override fun onDestroy() {
        super.onDestroy()
        if (bound) {
            discoveryService?.stopDiscovery()
            unbindService(connection)
            bound = false
        }
    }
}
```

### 3. Servers Adapter

```kotlin
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView

class ServersAdapter(
    private val onServerClick: (ServerDiscoveryService.ServerInfo) -> Unit
) : RecyclerView.Adapter<ServersAdapter.ServerViewHolder>() {
    
    private val servers = mutableListOf<ServerDiscoveryService.ServerInfo>()
    
    fun addServer(server: ServerDiscoveryService.ServerInfo) {
        if (!servers.any { it.ip == server.ip }) {
            servers.add(server)
            notifyItemInserted(servers.size - 1)
        }
    }
    
    fun clearServers() {
        servers.clear()
        notifyDataSetChanged()
    }
    
    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ServerViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(android.R.layout.simple_list_item_2, parent, false)
        return ServerViewHolder(view)
    }
    
    override fun onBindViewHolder(holder: ServerViewHolder, position: Int) {
        val server = servers[position]
        holder.bind(server)
    }
    
    override fun getItemCount() = servers.size
    
    inner class ServerViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        private val text1: TextView = itemView.findViewById(android.R.id.text1)
        private val text2: TextView = itemView.findViewById(android.R.id.text2)
        
        fun bind(server: ServerDiscoveryService.ServerInfo) {
            text1.text = server.name
            text2.text = "${server.ip}:${server.port} (v${server.version})"
            
            itemView.setOnClickListener {
                onServerClick(server)
            }
        }
    }
}
```

### 4. Layout File (activity_main.xml)

```xml
<?xml version="1.0" encoding="utf-8"?>
<LinearLayout xmlns:android="http://schemas.android.com/apk/res/android"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:orientation="vertical"
    android:padding="16dp">

    <TextView
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:text="🌳 Wood ID Server Discovery"
        android:textSize="24sp"
        android:textStyle="bold"
        android:gravity="center"
        android:layout_marginBottom="16dp" />

    <TextView
        android:id="@+id/statusText"
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:text="Ready to discover servers"
        android:textSize="16sp"
        android:gravity="center"
        android:layout_marginBottom="16dp" />

    <Button
        android:id="@+id/startDiscoveryButton"
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:text="Start Discovery"
        android:layout_marginBottom="16dp" />

    <TextView
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:text="Discovered Servers:"
        android:textSize="18sp"
        android:textStyle="bold"
        android:layout_marginBottom="8dp" />

    <androidx.recyclerview.widget.RecyclerView
        android:id="@+id/serversRecyclerView"
        android:layout_width="match_parent"
        android:layout_height="0dp"
        android:layout_weight="1" />

</LinearLayout>
```

### 5. Upload Activity Example

```kotlin
import android.os.Bundle
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import okhttp3.*
import java.io.File
import java.io.IOException

class UploadActivity : AppCompatActivity() {
    private lateinit var serverInfoText: TextView
    private lateinit var selectImageButton: Button
    private lateinit var uploadButton: Button
    private lateinit var imagePreview: ImageView
    
    private var selectedImageFile: File? = null
    private var serverIp: String = ""
    private var serverPort: Int = 5000
    
    private val client = OkHttpClient()
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_upload)
        
        serverInfoText = findViewById(R.id.serverInfoText)
        selectImageButton = findViewById(R.id.selectImageButton)
        uploadButton = findViewById(R.id.uploadButton)
        imagePreview = findViewById(R.id.imagePreview)
        
        // Get server info from intent
        serverIp = intent.getStringExtra("server_ip") ?: ""
        serverPort = intent.getIntExtra("server_port", 5000)
        val serverName = intent.getStringExtra("server_name") ?: "Unknown Server"
        
        serverInfoText.text = "Connected to: $serverName\n$serverIp:$serverPort"
        
        selectImageButton.setOnClickListener {
            // Implement image selection logic
            selectImage()
        }
        
        uploadButton.setOnClickListener {
            uploadImage()
        }
    }
    
    private fun selectImage() {
        // Implement image selection using Intent
        // This is a simplified example
    }
    
    private fun uploadImage() {
        selectedImageFile?.let { file ->
            val requestBody = MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("image", file.name,
                    RequestBody.create("image/*".toMediaTypeOrNull(), file))
                .addFormDataPart("class_name", "acacia_auriculiformis") // You can add a class selector
                .build()
            
            val request = Request.Builder()
                .url("http://$serverIp:$serverPort/upload")
                .post(requestBody)
                .build()
            
            client.newCall(request).enqueue(object : Callback {
                override fun onFailure(call: Call, e: IOException) {
                    runOnUiThread {
                        // Handle error
                    }
                }
                
                override fun onResponse(call: Call, response: Response) {
                    runOnUiThread {
                        if (response.isSuccessful) {
                            // Handle success
                        } else {
                            // Handle error
                        }
                    }
                }
            })
        }
    }
}
```

## Usage Instructions

1. **Start the Flask server** with UDP broadcasting:
   ```bash
   cd app
   python app.py
   ```

2. **In your Android app**:
   - Add the required permissions to `AndroidManifest.xml`
   - Implement the `ServerDiscoveryService`
   - Use the `MainActivity` to discover servers
   - Use the `UploadActivity` to upload images

3. **The discovery process**:
   - Android app listens on UDP port 5001
   - Flask server broadcasts every 5 seconds
   - When a broadcast is received, the server info is parsed and displayed
   - User can select a server to connect to

## Key Features

- **Automatic Discovery**: No need to manually enter server IP addresses
- **Real-time Updates**: Servers are discovered as they come online
- **Multiple Servers**: Can discover multiple servers on the network
- **Error Handling**: Robust error handling for network issues
- **Clean UI**: User-friendly interface for server selection

## Network Requirements

- Both devices must be on the same WiFi network
- Firewall must allow UDP traffic on port 5001
- Router must support UDP broadcast (most home routers do)

This implementation provides a seamless way for your Android app to automatically discover and connect to the Wood ID Upload Server! 