package com.bridgedeux.feature.translation.presentation.components

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.rounded.VolumeUp
import androidx.compose.material.icons.rounded.ArrowDropDown
import androidx.compose.material.icons.rounded.ContentCopy
import androidx.compose.material.icons.rounded.Save
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.bridgedeux.domain.model.Language

private val TargetCardShape = RoundedCornerShape(28.dp)

@Composable
fun TargetTranslationCard(
    selectedLanguage: Language,
    translatedText: String,
    onLanguageClick: () -> Unit,
    onCopyClick: () -> Unit,
    onSpeakClick: () -> Unit,
    onSaveClick: () -> Unit,
    modifier: Modifier = Modifier
) {

    Card(
        modifier = modifier.fillMaxWidth(),
        shape = TargetCardShape,
        elevation = CardDefaults.cardElevation(
            defaultElevation = 2.dp
        ),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surface
        )
    ) {

        LanguageHeader(
            language = selectedLanguage,
            onClick = onLanguageClick
        )

        HorizontalDivider()

        TranslationArea(
            translatedText = translatedText
        )

        HorizontalDivider()

        TargetToolbar(
            onCopyClick = onCopyClick,
            onSpeakClick = onSpeakClick,
            onSaveClick = onSaveClick
        )

    }

}

@Composable
private fun LanguageHeader(
    language: Language,
    onClick: () -> Unit
) {

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(
                onClick = onClick
            )
            .padding(
                horizontal = 20.dp,
                vertical = 16.dp
            ),
        verticalAlignment = Alignment.CenterVertically
    ) {

        Text(
            text = language.name
                .lowercase()
                .replaceFirstChar { it.titlecase() },
            style = MaterialTheme.typography.titleMedium,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis
        )

        Spacer(
            modifier = Modifier.width(4.dp)
        )

        Icon(
            imageVector = Icons.Rounded.ArrowDropDown,
            contentDescription = "Select target language"
        )

    }

}

@Composable
private fun TranslationArea(
    translatedText: String
) {

    Box(
        modifier = Modifier
            .fillMaxWidth()
            .heightIn(
                min = 140.dp
            )
            .padding(
                horizontal = 20.dp,
                vertical = 18.dp
            ),
        contentAlignment = Alignment.TopStart
    ) {

        if (translatedText.isBlank()) {

            Text(
                text = "Translation will appear here",
                style = MaterialTheme.typography.bodyLarge,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )

        } else {

            Text(
                text = translatedText,
                style = MaterialTheme.typography.bodyLarge,
                color = MaterialTheme.colorScheme.onSurface
            )

        }

    }

}

@Composable
private fun TargetToolbar(
    onCopyClick: () -> Unit,
    onSpeakClick: () -> Unit,
    onSaveClick: () -> Unit
) {

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(
                horizontal = 12.dp,
                vertical = 8.dp
            ),
        horizontalArrangement = Arrangement.End,
        verticalAlignment = Alignment.CenterVertically
    ) {

        IconButton(
            onClick = onCopyClick
        ) {

            Icon(
                imageVector = Icons.Rounded.ContentCopy,
                contentDescription = "Copy"
            )

        }

        Spacer(
            modifier = Modifier.width(4.dp)
        )

        IconButton(
            onClick = onSpeakClick
        ) {

            Icon(
                imageVector = Icons.AutoMirrored.Rounded.VolumeUp,
                contentDescription = "Speak"
            )

        }

        Spacer(
            modifier = Modifier.width(4.dp)
        )

        IconButton(
            onClick = onSaveClick
        ) {

            Icon(
                imageVector = Icons.Rounded.Save,
                contentDescription = "Save"
            )

        }

    }

}