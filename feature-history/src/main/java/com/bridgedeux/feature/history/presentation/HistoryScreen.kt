package com.bridgedeux.feature.history.presentation

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material3.Card
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.compose.foundation.lazy.items

@Composable
fun HistoryRoute(
    viewModel: HistoryViewModel
) {
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()

    HistoryScreen(
        uiState = uiState,
        onSearchQueryChanged = viewModel::onSearchQueryChanged
    )
}

@Composable
fun HistoryScreen(
    uiState: HistoryUiState,
    onSearchQueryChanged: (String) -> Unit,
    modifier: Modifier = Modifier
) {

    Column(
        modifier = modifier
            .fillMaxSize()
            .padding(24.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {

        Text(
            text = "History",
            style = MaterialTheme.typography.headlineMedium
        )

        OutlinedTextField(
            value = uiState.searchQuery,
            onValueChange = onSearchQueryChanged,
            modifier = Modifier.fillMaxWidth(),
            label = {
                Text("Search")
            }
        )

        when {

            uiState.isLoading -> {

                CircularProgressIndicator()

            }

            uiState.errorMessage != null -> {

                Text(
                    text = uiState.errorMessage,
                    color = MaterialTheme.colorScheme.error
                )

            }

            uiState.historyItems.isEmpty() -> {

                Text(
                    text = "No translations yet.",
                    style = MaterialTheme.typography.bodyLarge
                )

            }

            else -> {

                LazyColumn(
                    contentPadding = PaddingValues(vertical = 8.dp),
                    verticalArrangement = Arrangement.spacedBy(12.dp)
                ) {

                    items(uiState.historyItems) { item ->

                        Card(
                            modifier = Modifier.fillMaxWidth()
                        ) {

                            Column(
                                modifier = Modifier.padding(16.dp),
                                verticalArrangement = Arrangement.spacedBy(8.dp)
                            ) {

                                Text(
                                    text = "${item.sourceLanguage} → ${item.targetLanguage}",
                                    style = MaterialTheme.typography.labelLarge
                                )

                                Text(
                                    text = item.sourceText,
                                    style = MaterialTheme.typography.bodyLarge
                                )

                                Text(
                                    text = item.translatedText,
                                    style = MaterialTheme.typography.bodyMedium
                                )
                            }
                        }
                    }
                }
            }
        }
    }
}