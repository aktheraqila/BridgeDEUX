package com.bridgedeux.engine.mock.translation

import com.bridgedeux.core.result.AppResult
import com.bridgedeux.domain.model.Language
import com.bridgedeux.domain.model.TranslationRequest
import com.bridgedeux.domain.model.TranslationResult
import com.bridgedeux.domain.repository.TranslationRepository
import kotlinx.coroutines.delay
import kotlin.time.Duration.Companion.seconds

class MockTranslationRepository : TranslationRepository {

    override suspend fun translate(
        request: TranslationRequest
    ): AppResult<TranslationResult> {

        delay(1.seconds)

        val translatedText = when {
            request.sourceLanguage == Language.ENGLISH &&
                    request.targetLanguage == Language.GERMAN -> {
                translateEnglishToGerman(request.text)
            }

            request.sourceLanguage == Language.GERMAN &&
                    request.targetLanguage == Language.ENGLISH -> {
                translateGermanToEnglish(request.text)
            }

            else -> request.text
        }

        return AppResult.Success(
            TranslationResult(
                sourceText = request.text,
                translatedText = translatedText,
                sourceLanguage = request.sourceLanguage,
                targetLanguage = request.targetLanguage
            )
        )
    }

    private fun translateEnglishToGerman(
        text: String
    ): String {
        return when (text.lowercase()) {
            "hello" -> "Hallo"
            "good morning" -> "Guten Morgen"
            "how are you?" -> "Wie geht es dir?"
            else -> "[Mock DE] $text"
        }
    }

    private fun translateGermanToEnglish(
        text: String
    ): String {
        return when (text.lowercase()) {
            "hallo" -> "Hello"
            "guten morgen" -> "Good morning"
            "wie geht es dir?" -> "How are you?"
            else -> "[Mock EN] $text"
        }
    }
}