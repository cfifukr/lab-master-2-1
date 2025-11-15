pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Install dependencies') {
            steps {
                sh 'python3 -m venv venv'
                sh 'source venv/bin/activate && pip install -r requirements.txt'
            }
        }

        stage('Static Analysis - Pylint') {
            steps {
                sh 'source venv/bin/activate && pylint **/*.py > pylint-report.txt || true'
            }
        }

        stage('Run Tests') {
            steps {
                sh 'source venv/bin/activate && pip install pytest'
                sh 'source venv/bin/activate && pytest tests/ --junitxml=pytest-report.xml || true'
            }
            post {
                always {
                    junit 'pytest-report.xml'
                }
            }
        }
    }

    post {
        always {
            archiveArtifacts artifacts: 'pylint-report.txt', onlyIfSuccessful: true
        }
    }
}
