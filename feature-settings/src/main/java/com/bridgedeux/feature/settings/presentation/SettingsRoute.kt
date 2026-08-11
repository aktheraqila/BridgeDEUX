package com.bridgedeux.feature.settings.presentation

import androidx.compose.runtime.Composable
import androidx.lifecycle.compose.collectAsStateWithLifecycle

@Composable
fun SettingsRoute(
    viewModel: SettingsViewModel,
    onNavigateToAbout: () -> Unit
) {

    val uiState = viewModel
        .uiState
        .collectAsStateWithLifecycle()

    SettingsScreen(
        uiState = uiState.value,
        onThemeClicked = viewModel::onThemeClicked,
        onVoiceSettingsClicked = viewModel::onVoiceSettingsClicked,
        onOfflineModelsClicked = viewModel::onOfflineModelsClicked,
        onDeveloperModeClicked = viewModel::onDeveloperModeClicked,
        onAboutClicked = onNavigateToAbout
    )
}