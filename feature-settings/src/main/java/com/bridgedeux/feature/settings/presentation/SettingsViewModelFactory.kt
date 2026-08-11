package com.bridgedeux.feature.settings.presentation

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import com.bridgedeux.domain.repository.SettingsRepository

fun settingsViewModelFactory(
    settingsRepository: SettingsRepository
): ViewModelProvider.Factory =
    object : ViewModelProvider.Factory {

        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(
            modelClass: Class<T>
        ): T {

            require(
                modelClass.isAssignableFrom(
                    SettingsViewModel::class.java
                )
            ) {
                "Unknown ViewModel class: ${modelClass.name}"
            }

            return SettingsViewModel(
                settingsRepository = settingsRepository
            ) as T
        }
    }