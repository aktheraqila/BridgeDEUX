package com.bridgedeux.ui

import androidx.compose.material3.Scaffold
import androidx.compose.runtime.Composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import com.bridgedeux.AppContainer
import com.bridgedeux.navigation.BottomBar
import com.bridgedeux.navigation.BridgeDeuxNavHost

@Composable
fun BridgeDeuxAppRoot(
    appContainer: AppContainer
) {
    val navController = rememberNavController()

    val currentBackStackEntry =
        navController.currentBackStackEntryAsState()

    val currentRoute =
        currentBackStackEntry.value?.destination?.route

    Scaffold(
        bottomBar = {
            BottomBar(
                navController = navController,
                currentRoute = currentRoute
            )
        }
    ) { innerPadding ->

        BridgeDeuxNavHost(
            navController = navController,
            innerPadding = innerPadding,
            appContainer = appContainer
        )
    }
}