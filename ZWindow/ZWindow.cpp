#include "ZWindow.h"
#include <windowsx.h>
#include <iostream>
#include <thread>

static LRESULT CALLBACK Proc(HWND hwnd, UINT message, WPARAM wparam, LPARAM lparam)
{

	if (!ZLib::ZWindow::checkWindow(hwnd)) {
		return DefWindowProc(hwnd, message, wparam, lparam);
	}
	auto& config = ZLib::ZWindow::getWindow(hwnd).getConfig();
	return config.executeCallback(hwnd, message, wparam, lparam);
}



ZLib::ZWindow& ZLib::ZWindow::getWindow(HWND hwnd)
{
	return *windows[hwnd];
}

ZLib::ZWindow& ZLib::ZWindow::createWindow()
{
	auto ptr = std::shared_ptr<ZWindow>(new ZWindow());
	ptr->id_in_pool = windows_pool.size();
	windows_pool.push_back(ptr);
	return *ptr;
}

bool ZLib::ZWindow::checkWindow(HWND hwnd)
{
	return windows.find(hwnd)!=windows.end();
}

ZLib::ZWindow& ZLib::ZWindow::start()
{
	if (is_start) {
		return *this;
	}
	hwnd = CreateWindow(TEXT("MyClass"),    //��������
		config.window_name.c_str(),    //���ڱ��⣬���ڴ��ڵ����ϽǱ�������ʾ
		WS_OVERLAPPEDWINDOW | WS_CAPTION, //���ڷ��
		CW_USEDEFAULT,  //�������Ͻ�xλ�ã�����ʹ�õ�ϵͳĬ��ֵ�����Զ���
		CW_USEDEFAULT,  //�������Ͻ�yλ��
		config.width,  //���ڵĿ��
		config.height,  //���ڵĸ߶�
		NULL, //�ô��ڵĸ����ڻ������ߴ��ڵľ���������ò�������ΪNULL
		config.menus[""], //���ڲ˵����������û�в˵�������ΪNULL
		GetModuleHandle(NULL), //���ھ��
		NULL  //���ݸ�����WM_CREATE��Ϣ��һ�����������ﲻ�ã�����ΪNULL
	);
	windows[hwnd] = windows_pool[id_in_pool];
	BOOL success = ShowWindow(hwnd, SW_SHOW);
	success = UpdateWindow(hwnd);

	is_start = true;
	return *this;
}


WPARAM ZLib::ZWindow::loop()
{
	while ((!is_closed) && GetMessage(&msg, hwnd, NULL, 0)) {

		//������Ϣ
		TranslateMessage(&msg);
		//�ɷ���Ϣ
		DispatchMessage(&msg);
	}
	return 0;;
}

ZLib::WindowConfig& ZLib::ZWindow::getConfig()
{
	return config;
}

void ZLib::ZWindow::close()
{
	is_closed = true;
}

HWND ZLib::ZWindow::getHWND() const
{
	return hwnd;
}

bool ZLib::ZWindow::draw(const ImageBuffer & buffer, UINT message)
{
	if (buffer.getWidth() != config.width && buffer.getHeight() != config.height) {
		return false;
	}
	BITMAPINFOHEADER bi_header;
	memset(&bi_header, 0, sizeof(BITMAPINFOHEADER));
	bi_header.biSize = sizeof(BITMAPINFOHEADER);
	bi_header.biWidth = config.width;
	bi_header.biHeight = config.height;  
	bi_header.biPlanes = 1;
	bi_header.biBitCount = 32;
	bi_header.biCompression = BI_RGB;
	HDC hdc, hdcmem;
	PAINTSTRUCT pt;
	HBITMAP hbmp;
	u8* colors = nullptr;
	if (message == WM_PAINT) {
		hdc = BeginPaint(hwnd, &pt);
	} else {
		hdc = GetDC(hwnd);
	}
	
	hdcmem = CreateCompatibleDC(hdc);
	hbmp = CreateDIBSection(hdcmem, (BITMAPINFO*)&bi_header,DIB_RGB_COLORS, (void**)&colors, NULL, 0);
	memcpy(colors, buffer.getBuffer(), sizeof(Color) * config.width * config.height);
	auto old_bitmap = SelectObject(hdcmem, hbmp);
	BitBlt(hdc, 0, 0, config.width, config.height, hdcmem, 0, 0, SRCCOPY);
	if (message == WM_PAINT) {
		EndPaint(hwnd, &pt);
	} else {
		ReleaseDC(hwnd, hdc);
	}
	DeleteObject(old_bitmap);
	DeleteDC(hdcmem);
	DeleteObject(hbmp);
	
	return true;
}

void ZLib::ZWindow::init()
{
	CoInitializeEx(NULL, COINIT_MULTITHREADED);
	WNDCLASS wndclass;
	wndclass.style = CS_HREDRAW | CS_VREDRAW;
	wndclass.lpfnWndProc = Proc;
	wndclass.cbClsExtra = 0;
	wndclass.cbWndExtra = 0;
	wndclass.hInstance = GetModuleHandle(NULL);
	wndclass.hIcon = LoadIcon(NULL, IDI_APPLICATION);
	wndclass.hCursor = LoadCursor(NULL, IDC_ARROW);
	wndclass.hbrBackground = (HBRUSH)GetStockObject(WHITE_BRUSH);
	wndclass.lpszMenuName = NULL;
	wndclass.lpszClassName = TEXT("MyClass");

	if (!RegisterClass(&wndclass)) {
		//ע�ᴰ����ʧ��ʱ��������ʾ
		MessageBox(NULL, TEXT("This program requires Window NT!"), TEXT("MyClass"), MB_ICONERROR);
		exit(1);
	}
}


ZLib::ZWindow::ZWindow() :config(this), is_closed(false), is_start(false)
{


}




void ZLib::WindowConfig::appendMenuItem(const MenuInfo& info, const std::shared_ptr<Callback>& callback)
{
	auto menu = CreateMenu();
	menus[info.id_name] = menu;
	UINT uFlags = 0;
	LPCTSTR str = nullptr;
	std::shared_ptr<Callback> call = callback;
	UINT_PTR ptr = (UINT_PTR)menu;
	switch (info.type) {
	case MenuType::DropDown:
		uFlags = MF_STRING | MF_POPUP;
		str = info.display.c_str();
		break;
	case MenuType::Click:
		uFlags = MF_STRING;
		str = info.display.c_str();
		menu_callbacks[menu_id] = call;
		ptr = menu_id;
		++menu_id;
		break;
	default:
		break;
	}
	AppendMenuW(menus[info.father_name], uFlags, ptr, str);
	
}

void ZLib::WindowConfig::setTimerCallback(const std::shared_ptr<Callback>& callback, unsigned int timer_id, unsigned int t)
{
	timer_callbacks[timer_id] = callback;
	auto hwnd = owner->getHWND();
	SetTimer(hwnd, timer_id, t, NULL);

}

void ZLib::WindowConfig::setCallback(const std::shared_ptr<Callback>& callback, UINT message)
{
	message_callbacks[message] = callback;
}

LRESULT ZLib::WindowConfig::executeMenuCallback(HWND hwnd, UINT message, WPARAM wparam, LPARAM lparam)
{
	// �ж��Ƿ�Ϊ�˵�
	if (HIWORD(wparam) == 0) {
		auto id = LOWORD(wparam);
		auto iter = menu_callbacks.find(id);
		if (iter != menu_callbacks.end() && iter->second) {
			return (*iter->second)(hwnd, message, wparam, lparam);
		}
	}
	return USE_DEFAULT_CALLBACK;
}

LRESULT ZLib::WindowConfig::executeTimerCallback(HWND hwnd, UINT message, WPARAM wparam, LPARAM lparam)
{
	auto iter = timer_callbacks.find(wparam);
	if (iter!=timer_callbacks.end()) {
		return (*iter->second)(hwnd, message, wparam, lparam);
	}
	return USE_DEFAULT_CALLBACK;
}

LRESULT ZLib::WindowConfig::executeCallback(HWND hwnd, UINT message, WPARAM wparam, LPARAM lparam)
{
	LRESULT return_val = USE_DEFAULT_CALLBACK;
	switch (message) {
	case WM_SIZE:
		switch (wparam) {
		case SIZE_MINIMIZED:
			break;
		default:
			width = LOWORD(lparam);
			height = HIWORD(lparam);
			break;
		}
		break;
	case WM_TIMER:
		return_val = executeTimerCallback(hwnd, message, wparam, lparam);
		break;
	case WM_DESTROY:
		PostQuitMessage(0);
		owner->close();
		break;
	case WM_COMMAND:
		// �ж��Ƿ��ǲ˵�
		return_val = executeMenuCallback(hwnd, message, wparam, lparam);
		break;
	}
	if (return_val == USE_DEFAULT_CALLBACK) {
		auto iter = message_callbacks.find(message);
		if (iter != message_callbacks.end()) {
			return (*iter->second)(hwnd, message, wparam, lparam);
		}
		return DefWindowProc(hwnd, message, wparam, lparam);
	} else {
		return return_val;
	}
	
}

ZLib::WindowConfig::~WindowConfig()
{
	for (auto menu : menus) {
		DestroyMenu(menu.second);
	}
	
}

ZLib::WindowConfig::WindowConfig(ZWindow* ptr) :width(CW_USEDEFAULT), height(CW_USEDEFAULT), window_name(TEXT("NewWindow")),menu_id(0),owner(ptr)
{
	menus[""] = CreateMenu();

}

ZLib::WindowConfig::WindowConfig(const WindowConfig& config)
{}


ZLib::MenuInfo::MenuInfo(const std::string& id_name_, const std::wstring& display_str_, MenuType type_,const std::string& father_):
id_name(id_name_),display(display_str_),type(type_),father_name(father_)
{}

std::vector<std::wstring> ZLib::getFilePath(HWND hwnd, bool is_open,bool is_multi, const std::map<std::wstring, std::vector<std::wstring>>& filters)
{
	WCHAR str_filename[MAX_PATH];
	std::wstring filter_str;
	OPENFILENAME ofn;
	ZeroMemory(&ofn, sizeof(OPENFILENAME));
	ofn.hwndOwner = hwnd;
	ofn.lStructSize = sizeof(OPENFILENAME);
	ofn.lpstrFile = str_filename;
	ofn.lpstrFile[0] = TEXT('\0');
	ofn.nMaxFile = sizeof(str_filename);
	ofn.Flags = OFN_PATHMUSTEXIST | OFN_EXPLORER;
	ofn.hInstance= GetModuleHandle(NULL);
	ofn.lpstrInitialDir=L".\\";
	if (is_open) {
		ofn.Flags |= OFN_FILEMUSTEXIST;
	} else {
		ofn.Flags |= OFN_OVERWRITEPROMPT;
	}
	if (is_multi) {
		ofn.Flags |= OFN_ALLOWMULTISELECT;
	}
	if (filters.empty()) {
		ofn.lpstrFilter = NULL;
	} else{
		ofn.nFilterIndex = 1;
		for (auto& filter : filters) {
			filter_str += filter.first;
			filter_str.push_back(TEXT('\0'));
			bool first = true;
			for (auto& suffix : filter.second) {
				if (!first) {
					filter_str += TEXT(";");
				} else {
					first = false;
				}
				filter_str += TEXT("*.") + suffix;
			}
			filter_str.push_back(TEXT('\0'));
		}
		ofn.lpstrFilter = filter_str.c_str();
		if (!is_open) {
			bool flag = true;
			int c = 1;
			for (auto& filter : filters) {
				for (auto& suffix : filter.second) {
					if (suffix != L"*") {
						ofn.lpstrDefExt = suffix.c_str();
						flag = false;
						ofn.nFilterIndex = c;
						break;
					}
				}
				++c;
				if (!flag) {
					break;
				}
			}
		}
	}

	if (is_open) {
		//ShowWindow(hwnd, SW_HIDE);
		GetOpenFileName(&ofn);
		//ShowWindow(hwnd, SW_SHOW);
	} else {
		GetSaveFileName(&ofn);
	}
	std::vector<std::wstring> re;
	if (!is_multi) {
		re.push_back(std::wstring(str_filename));
	} else {
		std::wstring dir;
		int i;
		for (i = 0; str_filename[i] != L'\0'; ++i) {

		}
		dir= std::wstring(str_filename, str_filename + i);
		dir.push_back(L'\\');
		++i;
		for (; str_filename[i] != L'\0'; ++i) {
			int c = 0;
			while (str_filename[i + c] != L'\0') {
				++c;
			}
			re.push_back(dir + std::wstring(str_filename + i, str_filename + i + c));
			i += c;
		}
	}
	return re;
}

ZLib::__A::__A()
{
	ZWindow::init();
}

std::map<HWND, std::shared_ptr<ZLib::ZWindow>> ZLib::ZWindow::windows;
std::vector<std::shared_ptr<ZLib::ZWindow>> ZLib::ZWindow::windows_pool;
ZLib::__A ZLib::__A::a;