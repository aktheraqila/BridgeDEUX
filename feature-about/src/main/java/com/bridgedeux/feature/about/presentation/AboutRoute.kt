package com.bridgedeux.feature.about.presentation

import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.lifecycle.compose.collectAsStateWithLifecycle

@Composable
fun AboutRoute(
    viewModel: AboutViewModel
) {
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()

    AboutScreen(
        uiState = uiState
    )
}