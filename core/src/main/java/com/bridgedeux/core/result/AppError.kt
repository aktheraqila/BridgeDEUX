package com.bridgedeux.core.result

sealed interface AppError {

    data object Unknown : AppError

    data class InvalidInput(
        val reason: String
    ) : AppError

    data class ModelUnavailable(
        val modelId: String
    ) : AppError

    data class InferenceFailure(
        val message: String?
    ) : AppError
}