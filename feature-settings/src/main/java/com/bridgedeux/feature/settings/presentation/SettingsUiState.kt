package com.bridgedeux.feature.settings.presentation

data class SettingsUiState(

    val developerModeEnabled: Boolean = false,

    val appVersion: String = "1.0.0",

    val availableModels: Int = 0,

    val darkModeEnabled: Boolean = false,

    val voicePlaybackEnabled: Boolean = true
)