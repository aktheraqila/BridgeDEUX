package com.bridgedeux.navigation

import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Translate
import androidx.compose.material.icons.filled.History
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.Icon
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.NavigationBarItemDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBarDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.navigation.NavHostController
import androidx.navigation.navOptions

@Composable
fun BottomBar(
    navController: NavHostController,
    currentRoute: String?
) {

    NavigationBar(

        containerColor = MaterialTheme.colorScheme.surface,

        tonalElevation = NavigationBarDefaults.Elevation

    ) {

        Destination.bottomBarDestinations.forEach { destination ->

            NavigationBarItem(

                selected = currentRoute == destination.route,

                onClick = {

                    if (currentRoute == destination.route) return@NavigationBarItem

                    navController.navigate(
                        destination.route,
                        navOptions {

                            launchSingleTop = true

                            restoreState = true

                            popUpTo(
                                navController.graph.startDestinationId
                            ) {
                                saveState = true
                            }
                        }
                    )
                },

                icon = {

                    Icon(
                        imageVector = when (destination) {

                            Destination.Translate ->
                                Icons.Filled.Translate

                            Destination.History ->
                                Icons.Filled.History

                            Destination.Settings ->
                                Icons.Filled.Settings

                            else ->
                                Icons.Filled.Settings
                        },

                        contentDescription = destination.label
                    )
                },

                label = {
                    Text(destination.label)
                },

                alwaysShowLabel = true,

                colors = NavigationBarItemDefaults.colors(

                    selectedIconColor =
                        MaterialTheme.colorScheme.onPrimaryContainer,

                    selectedTextColor =
                        MaterialTheme.colorScheme.primary,

                    indicatorColor =
                        MaterialTheme.colorScheme.primaryContainer,

                    unselectedIconColor =
                        MaterialTheme.colorScheme.onSurfaceVariant,

                    unselectedTextColor =
                        MaterialTheme.colorScheme.onSurfaceVariant
                )
            )
        }
    }
}