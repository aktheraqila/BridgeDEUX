package com.bridgedeux.domain.usecase

import com.bridgedeux.core.result.AppError
import com.bridgedeux.core.result.AppResult
import com.bridgedeux.domain.model.Language
import com.bridgedeux.domain.model.TranslationRequest
import com.bridgedeux.domain.model.TranslationResult
import com.bridgedeux.domain.repository.TranslationRepository

class TranslateTextUseCase(
    private val translationRepository: TranslationRepository
) {

    suspend operator fun invoke(
        text: String,
        sourceLanguage: Language,
        targetLanguage: Language
    ): AppResult<TranslationResult> {

        val normalizedText = text.trim()

        if (normalizedText.isBlank()) {
            return AppResult.Error(
                AppError.InvalidInput(
                    reason = "Translation text cannot be empty."
                )
            )
        }

        if (sourceLanguage == targetLanguage) {
            return AppResult.Error(
                AppError.InvalidInput(
                    reason = "Source and target languages must be different."
                )
            )
        }

        val request = TranslationRequest(
            text = normalizedText,
            sourceLanguage = sourceLanguage,
            targetLanguage = targetLanguage
        )

        return translationRepository.translate(request)
    }
}