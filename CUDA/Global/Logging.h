#pragma once

#define logText(a,b)																					\
	if( (g_uiAllowedTypesConsole & (unsigned int)a)														\
		|| (g_uiAllowedTypesFile & (unsigned int)a) )													\
	{																									\
		Logging::logTextFileLine(a,b,__FILE__,__FUNCTION__,__LINE__);									\
	}


#define logTextParams(a,b,...)																			\
	if( (g_uiAllowedTypesConsole & (unsigned int)a)														\
		|| (g_uiAllowedTypesFile & (unsigned int)a) )													\
	{																									\
		Logging::logTextFileLine(a,""/*TiXmlString(b,__VA_ARGS__).c_str()*/,__FILE__,__FUNCTION__,__LINE__);	\
	}


#define logAssert(a)																					\
	if(a == 0)																					\
	{																									\
		logText(Logging::LT_ERROR,Str("Assert failed (exiting): "+Str(#a)).c_str());					\
		exit(1);																						\
	}

class Logging
{
public:
	enum LoggingType
	{	// all values must have different bits
		LT_INFORMATION	= 1 << 0,
		LT_WARNING		= 1 << 1,
		LT_ERROR		= 1 << 2,
		LT_DEBUG		= 1 << 3,
		LT_MEMORY		= 1 << 4
	};

	static void makeSureLoggingFileExists();

	static void logTextFileLine(LoggingType p_eLoggingType, const char *p_sLoggingText,const char *p_sFileName,const char *p_sFunctionName,long p_lLineNumber);

	static void setAllowedLoggingTypes(unsigned int p_uiNewAllowedTypesConsole, unsigned int p_uiNewAllowedTypesFile);

	class Timer
	{
	#ifdef _WIN32
		unsigned long m_ulStart;
	#else
		timespec m_Start;
		timespec m_Stop;
	#endif
	unsigned long m_ulTime;

	public:
		void start()
		{
		#ifdef _WIN32
			m_ulStart = timeGetTime ();
		#else
			do
			{
				clock_gettime(CLOCK_REALTIME, &m_Start);
			}
			while (m_Start.tv_nsec < 0 || m_Start.tv_nsec >= 1000000000L);
		#endif
		}

		unsigned long stop()
		{
		#ifdef _WIN32
			m_ulTime = timeGetTime() - m_ulStart;
			return m_ulTime;
		#else
			do
			{
				clock_gettime(CLOCK_REALTIME, &m_Stop);
			}
			while (m_Stop.tv_nsec < 0 || m_Stop.tv_nsec >= 1000000000L);

			m_ulTime = (m_Stop.tv_sec - m_Start.tv_sec)*1000 + (m_Stop.tv_nsec - m_Start.tv_nsec) / 1000000L;
			return m_ulTime;
		#endif
		}

		unsigned long getTime()
		{
			return m_ulTime;
		}
	};

private:
	static FILE *m_pLoggingFile;
};

// these variables should be inside Logging class, but they are used by kernels in TrainNetwork.cu, so they have to be global
extern unsigned int g_uiAllowedTypesConsole;
extern unsigned int g_uiAllowedTypesFile;
