//package com.bridgedeux.feature.settings.presentation
//
//import androidx.compose.foundation.clickable
//import androidx.compose.foundation.layout.Arrangement
//import androidx.compose.foundation.layout.Column
//import androidx.compose.foundation.layout.PaddingValues
//import androidx.compose.foundation.layout.fillMaxSize
//import androidx.compose.foundation.layout.fillMaxWidth
//import androidx.compose.foundation.layout.padding
//import androidx.compose.foundation.lazy.LazyColumn
//import androidx.compose.foundation.lazy.items
//import androidx.compose.material3.Card
//import androidx.compose.material3.CardDefaults
//import androidx.compose.material3.MaterialTheme
//import androidx.compose.material3.Scaffold
//import androidx.compose.material3.Text
//import androidx.compose.material3.TopAppBar
//import androidx.compose.material3.ExperimentalMaterial3Api
//import androidx.compose.runtime.Composable
//import androidx.compose.ui.Modifier
//import androidx.compose.ui.unit.dp
//
//@OptIn(ExperimentalMaterial3Api::class)
//@Composable
//fun SettingsScreen(
//    uiState: SettingsUiState,
//    onThemeClicked: () -> Unit,
//    onVoiceSettingsClicked: () -> Unit,
//    onOfflineModelsClicked: () -> Unit,
//    onDeveloperModeClicked: () -> Unit,
//    onAboutClicked: () -> Unit
//) {
//
//    val settingsItems = listOf(
//        "Appearance" to onThemeClicked,
//        "Voice Settings" to onVoiceSettingsClicked,
//        "Offline Models" to onOfflineModelsClicked,
//        "Developer Mode" to onDeveloperModeClicked,
//        "About" to onAboutClicked
//    )
//
//    Scaffold(
//        topBar = {
//            TopAppBar(
//                title = {
//                    Text("Settings")
//                }
//            )
//        }
//    ) { padding ->
//
//        LazyColumn(
//            modifier = Modifier
//                .fillMaxSize()
//                .padding(padding),
//            contentPadding = PaddingValues(16.dp),
//            verticalArrangement = Arrangement.spacedBy(12.dp)
//        ) {
//
//            item {
//                Text(
//                    text = "BridgeDEUX",
//                    style = MaterialTheme.typography.headlineSmall
//                )
//            }
//
//            item {
//                Text(
//                    text = "Version ${uiState.appVersion}",
//                    style = MaterialTheme.typography.bodyMedium
//                )
//            }
//
//            items(settingsItems) { (title, action) ->
//
//                Card(
//                    modifier = Modifier
//                        .fillMaxWidth()
//                        .clickable {
//                            action()
//                        },
//                    elevation = CardDefaults.cardElevation(
//                        defaultElevation = 2.dp
//                    )
//                ) {
//
//                    Column(
//                        modifier = Modifier.padding(20.dp)
//                    ) {
//
//                        Text(
//                            text = title,
//                            style = MaterialTheme.typography.titleMedium
//                        )
//
//                    }
//
//                }
//
//            }
//
//        }
//
//    }
//
//}

package com.bridgedeux.feature.settings.presentation

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

@OptIn(androidx.compose.material3.ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(
    uiState: SettingsUiState,
    onThemeClicked: () -> Unit,
    onVoiceSettingsClicked: () -> Unit,
    onOfflineModelsClicked: () -> Unit,
    onDeveloperModeClicked: () -> Unit,
    onAboutClicked: () -> Unit
) {

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Text("Settings")
                }
            )
        }
    ) { padding ->

        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding),
            contentPadding = PaddingValues(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {

            item {
                Text(
                    text = "BridgeDEUX",
                    style = MaterialTheme.typography.headlineSmall
                )
            }

            item {
                Text(
                    text = "Version ${uiState.appVersion}",
                    style = MaterialTheme.typography.bodyMedium
                )
            }

            item {
                SettingsSwitchCard(
                    title = "Appearance",
                    description = "Dark mode",
                    checked = uiState.darkModeEnabled,
                    onCheckedChange = {
                        onThemeClicked()
                    }
                )
            }

            item {
                SettingsSwitchCard(
                    title = "Voice Settings",
                    description = "Voice playback",
                    checked = uiState.voicePlaybackEnabled,
                    onCheckedChange = {
                        onVoiceSettingsClicked()
                    }
                )
            }

            item {
                SettingsNavigationCard(
                    title = "Offline Models",
                    onClick = onOfflineModelsClicked
                )
            }

            item {
                SettingsNavigationCard(
                    title = "Developer Mode",
                    onClick = onDeveloperModeClicked
                )
            }

            item {
                SettingsNavigationCard(
                    title = "About",
                    onClick = onAboutClicked
                )
            }
        }
    }
}

@Composable
private fun SettingsSwitchCard(
    title: String,
    description: String,
    checked: Boolean,
    onCheckedChange: () -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        elevation = CardDefaults.cardElevation(
            defaultElevation = 2.dp
        )
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(20.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {

            Column(
                modifier = Modifier.weight(1f)
            ) {
                Text(
                    text = title,
                    style = MaterialTheme.typography.titleMedium
                )

                Text(
                    text = description,
                    style = MaterialTheme.typography.bodyMedium
                )
            }

            Switch(
                checked = checked,
                onCheckedChange = {
                    onCheckedChange()
                }
            )
        }
    }
}

@Composable
private fun SettingsNavigationCard(
    title: String,
    onClick: () -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onClick),
        elevation = CardDefaults.cardElevation(
            defaultElevation = 2.dp
        )
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(20.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = title,
                style = MaterialTheme.typography.titleMedium
            )
        }
    }
}