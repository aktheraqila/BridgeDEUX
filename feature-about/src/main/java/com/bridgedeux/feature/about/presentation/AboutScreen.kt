package com.bridgedeux.feature.about.presentation

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material3.CenterAlignedTopAppBar
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedCard
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AboutScreen(
    uiState: AboutUiState,
    modifier: Modifier = Modifier
) {
    Scaffold(
        modifier = modifier,
        topBar = { AboutTopBar() }
    ) { innerPadding ->
        AboutContent(uiState, innerPadding)
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun AboutTopBar() {
    CenterAlignedTopAppBar(
        title = { Text("About", style = MaterialTheme.typography.titleLarge) }
    )
}

@Composable
private fun AboutContent(
    uiState: AboutUiState,
    contentPadding: PaddingValues
) {
    LazyColumn(
        modifier = Modifier.fillMaxSize().padding(contentPadding),
        contentPadding = PaddingValues(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        item { ApplicationCard(uiState) }
        item { ResearchCard(uiState) }
        item { AcademicCard(uiState) }
        item { LicenseCard(uiState) }
    }
}

@Composable
private fun ApplicationCard(uiState: AboutUiState) {
    OutlinedCard(modifier = Modifier.fillMaxWidth()) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            SectionTitle("Application")
            Text(uiState.appName, style = MaterialTheme.typography.headlineSmall)
            Text("Version ${uiState.version}", style = MaterialTheme.typography.bodyMedium)
            Text(uiState.description, style = MaterialTheme.typography.bodyLarge)
        }
    }
}

@Composable
private fun ResearchCard(uiState: AboutUiState) {
    OutlinedCard(modifier = Modifier.fillMaxWidth()) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            SectionTitle("Research")
            Text(uiState.researchTitle, style = MaterialTheme.typography.titleMedium)
            Text(uiState.description, style = MaterialTheme.typography.bodyMedium)
        }
    }
}

@Composable
private fun AcademicCard(uiState: AboutUiState) {
    OutlinedCard(modifier = Modifier.fillMaxWidth()) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            SectionTitle("Academic")
            InfoRow("Supervisor", uiState.supervisor)
            HorizontalDivider()
            InfoRow("Department", uiState.department)
            HorizontalDivider()
            InfoRow("University", uiState.university)
        }
    }
}

@Composable
private fun LicenseCard(uiState: AboutUiState) {
    OutlinedCard(modifier = Modifier.fillMaxWidth()) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            SectionTitle("License")
            Text(uiState.license, style = MaterialTheme.typography.bodyMedium)
        }
    }
}

@Composable
private fun SectionTitle(title: String) {
    Text(title, style = MaterialTheme.typography.titleMedium)
}

@Composable
private fun InfoRow(label: String, value: String) {
    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
        Text(label, style = MaterialTheme.typography.labelMedium)
        Text(value, style = MaterialTheme.typography.bodyLarge)
    }
}
