package com.bridgedeux.feature.about.presentation

import androidx.lifecycle.ViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * ViewModel for the About feature.
 *
 * Responsible only for exposing immutable About screen information
 * to the presentation layer.
 */
class AboutViewModel : ViewModel() {

    private val _uiState = MutableStateFlow(
        AboutUiState(
            appName = "BridgeDEUX",
            version = "1.0.0",

            researchTitle = "Adaptive German-to-English and English-to-German Translation on Mobile Devices",
            description = "An offline Android application for adaptive bidirectional speech and text translation using on-device AI.",

            supervisor = "Dr. Tanvir Azshar",
            department = "Department of Computer Science and Engineering",
            university = "East Delta University",

            license = "© 2026 BridgeDEUX Project. All rights reserved."
        )
    )

    val uiState: StateFlow<AboutUiState> =
        _uiState.asStateFlow()
}