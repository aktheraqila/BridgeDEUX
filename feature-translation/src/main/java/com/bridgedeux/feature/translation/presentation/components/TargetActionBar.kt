package com.bridgedeux.feature.translation.presentation.components

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.VolumeUp
import androidx.compose.material.icons.filled.ContentCopy
import androidx.compose.material.icons.filled.Save
import androidx.compose.material3.Icon
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier

@Composable
fun TargetActionBar(
    onCopyClick: () -> Unit,
    onSpeakClick: () -> Unit,
    onSaveClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    Row(
        modifier = modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween
    ) {

        OutlinedButton(
            onClick = onCopyClick
        ) {
            Icon(
                imageVector = Icons.Default.ContentCopy,
                contentDescription = "Copy"
            )
            Text(" Copy")
        }

        OutlinedButton(
            onClick = onSpeakClick
        ) {
            Icon(
                imageVector = Icons.AutoMirrored.Filled.VolumeUp,
                contentDescription = "Speak"
            )
            Text(" Speak")
        }

        OutlinedButton(
            onClick = onSaveClick
        ) {
            Icon(
                imageVector = Icons.Default.Save,
                contentDescription = "Save"
            )
            Text(" Save")
        }
    }
}