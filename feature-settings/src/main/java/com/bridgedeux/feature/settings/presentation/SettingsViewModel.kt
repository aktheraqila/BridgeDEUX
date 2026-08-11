package com.bridgedeux.feature.settings.presentation

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.bridgedeux.domain.repository.SettingsRepository
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.launch

class SettingsViewModel(
    private val settingsRepository: SettingsRepository
) : ViewModel() {

    private val _uiState = MutableStateFlow(
        SettingsUiState()
    )

    val uiState: StateFlow<SettingsUiState> =
        _uiState.asStateFlow()

    init {
        observeSettings()
    }

    private fun observeSettings() {
        viewModelScope.launch {
            combine(
                settingsRepository.observeDeveloperMode(),
                settingsRepository.observeDarkMode(),
                settingsRepository.observeVoicePlayback()
            ) { developerMode, darkMode, voicePlayback ->

                _uiState.value.copy(
                    developerModeEnabled = developerMode,
                    darkModeEnabled = darkMode,
                    voicePlaybackEnabled = voicePlayback
                )

            }.collect { state ->
                _uiState.value = state
            }
        }
    }

    fun onThemeClicked() {
        viewModelScope.launch {
            val currentValue = _uiState.value.darkModeEnabled

            settingsRepository.setDarkModeEnabled(
                enabled = !currentValue
            )
        }
    }

    fun onVoiceSettingsClicked() {
        viewModelScope.launch {
            val currentValue = _uiState.value.voicePlaybackEnabled

            settingsRepository.setVoicePlaybackEnabled(
                enabled = !currentValue
            )
        }
    }

    fun onOfflineModelsClicked() {
        // Phase 4 (ONNX Model Management)
    }

    fun onDeveloperModeClicked() {
        viewModelScope.launch {
            val currentValue = _uiState.value.developerModeEnabled

            settingsRepository.setDeveloperModeEnabled(
                enabled = !currentValue
            )
        }
    }

    fun onAboutClicked() {
        // Future feature-about module
    }
}