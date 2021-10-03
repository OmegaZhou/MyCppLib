#pragma once
#include <windows.h>
#include <string>
#include <map>
#include <memory>
#include <vector>
#include "../ImageTool/ImageBuffer.hpp"
#define DRAW_TIME_ID 1027
#define FRAME_RATE 30
#define USE_DEFAULT_CALLBACK 1998
namespace ZLib
{
	class ZWindow;
	enum class MenuType
	{
		DropDown,
		Click
	};
	struct MenuInfo
	{
		MenuInfo(const std::string& id_name,const std::wstring& display_str, MenuType type,const std::string& father = "");
		std::string father_name;
		std::string id_name;
		std::wstring display;
		MenuType type;
	};

	class Callback
	{
	public:
		virtual int operator()(HWND hwnd, UINT message, WPARAM wparam, LPARAM lparam)
		{
			return 0;
		};
	};
	std::vector<std::wstring> getFilePath(HWND hwnd, bool is_select,bool is_multi=false, const std::map<std::wstring, std::vector<std::wstring>>& filter = {});

	struct WindowConfig
	{
		friend class ZWindow;
		int width, height;
		std::wstring window_name;
		void appendMenuItem(const MenuInfo& info, const std::shared_ptr<Callback>& callback);
		void setTimerCallback(const std::shared_ptr<Callback>& callback, unsigned int timer_id, unsigned int t = 1000 / FRAME_RATE);
		void setCallback(const std::shared_ptr<Callback>& callback, UINT message);
		LRESULT executeMenuCallback(HWND hwnd, UINT message, WPARAM wparam, LPARAM lparam);
		LRESULT executeTimerCallback(HWND hwnd, UINT message, WPARAM wparam, LPARAM lparam);
		LRESULT executeCallback(HWND hwnd, UINT message, WPARAM wparam, LPARAM lparam);
		~WindowConfig();
	private:
		ZWindow* owner;
		unsigned int menu_id;
		WindowConfig(ZWindow* ptr);
		WindowConfig(const WindowConfig& config);
		std::map<std::string, HMENU> menus;
		std::map<unsigned int, std::shared_ptr<Callback>> menu_callbacks;
		std::map<unsigned int, std::shared_ptr<Callback>> timer_callbacks;
		std::map<UINT, std::shared_ptr<Callback>> message_callbacks;
	};

	class ZWindow
	{
		friend class __A;
	public:
		static ZWindow& getWindow(HWND hwnd);
		static ZWindow& createWindow();
		static bool checkWindow(HWND hwnd);
		ZWindow& start();
		WPARAM loop();
		WindowConfig& getConfig();
		void close();
		HWND getHWND()const;
		bool draw(const ImageBuffer & buffer, UINT message);
	private:
		static void init();
		static std::map<HWND, std::shared_ptr<ZWindow>> windows;
		static std::vector<std::shared_ptr<ZWindow>> windows_pool;
		ZWindow();
		ZWindow(const ZWindow& window) = delete;
		ZWindow& operator=(const ZWindow& window) = delete;
		HWND hwnd;
		MSG msg;
		WindowConfig config;
		bool is_closed;
		size_t id_in_pool;
		bool is_start;
	};
	class __A
	{
	private:
		static __A a;
		__A();
	};
}


