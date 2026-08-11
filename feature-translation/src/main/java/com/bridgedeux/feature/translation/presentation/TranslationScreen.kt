package com.bridgedeux.feature.translation.presentation

import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import com.bridgedeux.feature.translation.presentation.components.FloatingSwapButton
import com.bridgedeux.feature.translation.presentation.components.PrimaryTranslateButton
import com.bridgedeux.feature.translation.presentation.components.SourceTranslationCard
import com.bridgedeux.feature.translation.presentation.components.TargetTranslationCard
import com.bridgedeux.feature.translation.presentation.components.TranslationHeader
import androidx.compose.runtime.getValue
import androidx.lifecycle.compose.collectAsStateWithLifecycle

@Composable
fun TranslationRoute(
    viewModel: TranslationViewModel
) {
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()

    TranslationScreen(
        uiState = uiState,
        onInputTextChanged = viewModel::onInputTextChanged,
        onSwapLanguages = viewModel::onSwapLanguages,
        onTranslateClicked = viewModel::onTranslateClicked
    )
}

@Composable
fun TranslationScreen(
    uiState: TranslationUiState,
    onInputTextChanged: (String) -> Unit,
    onSwapLanguages: () -> Unit,
    onTranslateClicked: () -> Unit,
    modifier: Modifier = Modifier
) {

    val scrollState = rememberScrollState()

    Column(
        modifier = modifier
            .fillMaxSize()
            .verticalScroll(scrollState)
            .padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(24.dp)
    ) {

        TranslationHeader()

        Box(
            modifier = Modifier.fillMaxWidth()
        ) {

            Column(
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {

                SourceTranslationCard(
                    selectedLanguage = uiState.sourceLanguage,
                    inputText = uiState.inputText,
                    isEnabled = !uiState.isLoading,
                    onLanguageClick = {
                        // Phase 8
                    },
                    onTextChanged = onInputTextChanged,
                    onMicrophoneClick = {
                        // Phase 4
                    },
                    onClearClick = {
                        // Phase 2
                    }
                )

                TargetTranslationCard(
                    selectedLanguage = uiState.targetLanguage,
                    translatedText = uiState.translatedText,
                    onLanguageClick = {
                        // Phase 8
                    },
                    onCopyClick = {
                        // Phase 9
                    },
                    onSpeakClick = {
                        // Phase 9
                    },
                    onSaveClick = {
                        // Phase 3
                    }
                )
            }

            FloatingSwapButton(
                enabled = !uiState.isLoading,
                onClick = onSwapLanguages,
                modifier = Modifier.align(Alignment.Center).padding(top = 24.dp)
            )
        }

        PrimaryTranslateButton(
            isLoading = uiState.isLoading,
            enabled = !uiState.isLoading,
            onClick = onTranslateClicked
        )

        uiState.errorMessage?.let { error ->

            Card(
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.errorContainer
                )
            ) {

                Text(
                    text = error,
                    modifier = Modifier.padding(16.dp),
                    color = MaterialTheme.colorScheme.onErrorContainer
                )
            }
        }
    }
}