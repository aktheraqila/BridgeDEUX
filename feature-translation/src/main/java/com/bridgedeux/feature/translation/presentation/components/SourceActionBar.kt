package com.bridgedeux.feature.translation.presentation.components

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Clear
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.SwapHoriz
import androidx.compose.material3.Icon
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

@Composable
fun SourceActionBar(
    onMicrophoneClick: () -> Unit,
    onSwapLanguages: () -> Unit,
    onClearClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    Row(
        modifier = modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween
    ) {

        OutlinedButton(
            onClick = onMicrophoneClick
        ) {
            Icon(
                imageVector = Icons.Default.Mic,
                contentDescription = "Microphone"
            )
            Text(" Mic")
        }

        OutlinedButton(
            onClick = onSwapLanguages
        ) {
            Icon(
                imageVector = Icons.Default.SwapHoriz,
                contentDescription = "Swap languages"
            )
            Text(" Swap")
        }

        OutlinedButton(
            onClick = onClearClick
        ) {
            Icon(
                imageVector = Icons.Default.Clear,
                contentDescription = "Clear"
            )
            Text(" Clear")
        }
    }
}