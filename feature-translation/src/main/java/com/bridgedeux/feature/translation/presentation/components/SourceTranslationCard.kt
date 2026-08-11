package com.bridgedeux.feature.translation.presentation.components

import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.defaultMinSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.ArrowDropDown
import androidx.compose.material.icons.rounded.Clear
import androidx.compose.material.icons.rounded.Mic
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.DividerDefaults
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LocalTextStyle
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.bridgedeux.domain.model.Language
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.role
import androidx.compose.ui.semantics.semantics

private val CardShape = RoundedCornerShape(28.dp)

private const val MAX_CHARACTERS = 4000

@Composable
fun SourceTranslationCard(
    selectedLanguage: Language,
    inputText: String,
    isEnabled: Boolean,
    onLanguageClick: () -> Unit,
    onTextChanged: (String) -> Unit,
    onMicrophoneClick: () -> Unit,
    onClearClick: () -> Unit,
    modifier: Modifier = Modifier
) {

    Card(
        modifier = modifier.fillMaxWidth(),
        shape = CardShape,
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

        HorizontalDivider(Modifier, DividerDefaults.Thickness, DividerDefaults.color)

        EditorArea(
            value = inputText,
            enabled = isEnabled,
            onValueChange = { newValue ->

                if (newValue.length <= MAX_CHARACTERS) {
                    onTextChanged(newValue)
                }

            }
        )

        HorizontalDivider(Modifier, DividerDefaults.Thickness, DividerDefaults.color)

        BottomToolbar(
            characterCount = inputText.length,
            onMicClick = onMicrophoneClick,
            onClearClick = onClearClick,
            showClear = inputText.isNotBlank()
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
                indication = null,
                interactionSource = remember {
                    MutableInteractionSource()
                }
            ) {
                onClick()
            }
            .padding(
                horizontal = 20.dp,
                vertical = 16.dp
            ),
        verticalAlignment = Alignment.CenterVertically
    ) {

        Text(
            text = language.name,
            style = MaterialTheme.typography.titleMedium,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis
        )

        Spacer(Modifier.width(4.dp))

        Icon(
            imageVector = Icons.Rounded.ArrowDropDown,
            contentDescription = null
        )
    }
}

@Composable
private fun EditorArea(
    value: String,
    enabled: Boolean,
    onValueChange: (String) -> Unit
) {

    BasicTextField(
        value = value,
        onValueChange = onValueChange,
        enabled = enabled,
        textStyle = LocalTextStyle.current.copy(
            color = MaterialTheme.colorScheme.onSurface
        ),
        cursorBrush = SolidColor(
            MaterialTheme.colorScheme.primary
        ),
        keyboardOptions = KeyboardOptions(
            imeAction = ImeAction.Default
        ),
        keyboardActions = KeyboardActions(),
        modifier = Modifier
            .fillMaxWidth()
            .defaultMinSize(
                minHeight = 140.dp
            )
            .padding(
                horizontal = 20.dp,
                vertical = 18.dp
            ),
        decorationBox = { innerTextField ->

            Box(
                modifier = Modifier.fillMaxWidth()
            ) {

                if (value.isEmpty()) {

                    Text(
                        text = "Enter text",
                        style = MaterialTheme.typography.bodyLarge,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )

                }

                innerTextField()

            }

        }
    )
}

@Composable
private fun BottomToolbar(
    characterCount: Int,
    onMicClick: () -> Unit,
    onClearClick: () -> Unit,
    showClear: Boolean
) {

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(
                horizontal = 12.dp,
                vertical = 8.dp
            ),
        verticalAlignment = Alignment.CenterVertically
    ) {

        CharacterCounter(
            current = characterCount,
            maximum = MAX_CHARACTERS
        )

        Spacer(modifier = Modifier.weight(1f))

        IconButton(
            modifier = Modifier.semantics {
                contentDescription = "Start voice input"
                role = Role.Button
            },
            onClick = onMicClick
        ) {
            Icon(
                imageVector = Icons.Rounded.Mic,
                contentDescription = null
            )
        }

        if (showClear) {

            IconButton(
                modifier = Modifier.semantics {
                    contentDescription = "Clear input text"
                    role = Role.Button
                },
                onClick = onClearClick
            ) {
                Icon(
                    imageVector = Icons.Rounded.Clear,
                    contentDescription = null
                )
            }

        } else {

            Spacer(
                modifier = Modifier.size(48.dp)
            )

        }

    }

}

@Composable
private fun CharacterCounter(
    current: Int,
    maximum: Int
) {

    Text(
        text = "$current / $maximum",
        style = MaterialTheme.typography.labelMedium,
        color = MaterialTheme.colorScheme.onSurfaceVariant
    )

}