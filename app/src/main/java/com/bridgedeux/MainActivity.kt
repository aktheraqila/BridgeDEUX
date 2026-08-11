package com.bridgedeux

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import com.bridgedeux.ui.BridgeDeuxAppRoot
import com.bridgedeux.ui.theme.BridgeDEUXTheme

class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val appContainer =
            (application as BridgeDeuxApp).appContainer

        setContent {
            BridgeDEUXTheme {
                BridgeDeuxAppRoot(
                    appContainer = appContainer
                )
            }
        }
    }
}