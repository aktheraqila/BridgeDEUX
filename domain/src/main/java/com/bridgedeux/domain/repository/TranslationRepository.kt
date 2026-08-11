package com.bridgedeux.domain.repository

import com.bridgedeux.core.result.AppResult
import com.bridgedeux.domain.model.TranslationRequest
import com.bridgedeux.domain.model.TranslationResult

interface TranslationRepository {

    suspend fun translate(
        request: TranslationRequest
    ): AppResult<TranslationResult>
}